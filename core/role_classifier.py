"""
Role Classifier for Legal Document Chunks using fine-tuned BERT.

Uses a fine-tuned BertForSequenceClassification model to classify legal document
chunks into roles: Arguments, Precedents, Facts, Issues, Reasoning, Decision,
Statute, Preamble, Others.

Load via: RoleClassifier.load(model_dir)  or  create_classifier_from_config()

Changes vs original (v2):
    FIX-A  _load_model: DataParallel REMOVED entirely. DataParallel uses NCCL
           for cross-GPU scatter/gather which requires the *same OS thread* to
           own the CUDA context throughout the lifetime of every forward pass.
           Our pipeline runs stage-5 in a dedicated thread that is consistent,
           but NCCL still fails because the master thread that built the NCCL
           communicator (the constructor thread) is different from the inference
           thread. The correct multi-GPU strategy for a batch-inference pipeline
           is to run on a single dedicated GPU (cuda:N) and let the chunker's
           encode_batch occupy a different GPU if desired — throughput comes
           from large batch sizes, not data parallelism across GPUs.

    FIX-B  _load_model: added torch.compile() opt-in (PyTorch ≥ 2.0) behind
           a config flag so users who want extra throughput can enable it
           without touching this file.

    FIX-C  predict: removed the reactive NCCL fallback try/except block —
           it is no longer needed and was masking the real error for 1+ batches
           before triggering (causing partial-batch result corruption).

    FIX-D  predict: pin_memory() is only valid for CPU tensors being moved to
           CUDA. Added guard so it is skipped when device is CPU (avoids a
           silent RuntimeError on CPU-only machines).

    FIX-E  save: unwrap any accidental DataParallel wrapper before calling
           save_pretrained so the saved checkpoint is always a plain model.
"""
import contextlib
import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import json
from pathlib import Path
from tqdm import tqdm
from config import ROLE_CLASSIFICATION_CONFIG

logger = logging.getLogger(__name__)


class ChunkDataset(Dataset):
    """PyTorch Dataset for chunk classification."""

    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer=None, max_length=512):
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class RoleClassifier:
    """
    Legal document chunk role classifier using a fine-tuned transformer model.
    Supports inference and fine-tuning on custom role definitions.
    """

    def __init__(
        self,
        role_definitions: List[str],
        model_name: str = "distilbert-base-uncased",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        self.role_definitions = role_definitions
        self.num_labels = len(role_definitions)
        self.max_length = max_length

        self.label2id = {role: idx for idx, role in enumerate(role_definitions)}
        self.id2label = {idx: role for idx, role in enumerate(role_definitions)}

        # ----------------------------------------------------------------
        # FIX-A: device selection — pick a single GPU for the classifier.
        # If the config specifies a device (e.g. "cuda:1") use that so the
        # classifier and the embedding model can be pinned to different GPUs.
        # Default falls back to cuda:0, then cpu.
        # ----------------------------------------------------------------
        if device is None:
            config_device = ROLE_CLASSIFICATION_CONFIG.get("device", None)
            if config_device:
                self.device = torch.device(config_device)
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._use_amp     = ROLE_CLASSIFICATION_CONFIG.get("use_amp", True)
        self._num_workers = ROLE_CLASSIFICATION_CONFIG.get("num_workers", 4)

        logger.info(f"Initializing RoleClassifier with {self.num_labels} roles on {self.device}")

        self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load tokenizer and model onto a single device.

        FIX-A: DataParallel (and NCCL) are intentionally NOT used here.
        DataParallel requires the NCCL communicator to be owned by the thread
        that performs every forward pass. Because stage-5 of the production
        pipeline runs in a dedicated daemon thread that is *different* from the
        constructor thread, NCCL raises "unhandled cuda error" on the very first
        batch. Single-device inference on a large batch (CHUNK_BATCH=2048) on
        an A100 saturates the GPU just as well without any cross-device overhead.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.role_definitions)
            )
            logger.info(f"Loaded model from: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

        # Move to the single chosen device — no DataParallel wrapper
        self.model.to(self.device)
        self.model.eval()

        available = torch.cuda.device_count()
        # FIX-A: log which GPUs are available but make clear we are using one
        if available > 1:
            logger.info(
                f"RoleClassifier: {available} GPUs detected — using single device "
                f"{self.device} (DataParallel disabled; incompatible with threaded pipeline). "
                f"To use a different GPU set ROLE_CLASSIFICATION_CONFIG['device'] = 'cuda:N'."
            )
        else:
            logger.info(f"RoleClassifier: using device {self.device}")

        # FIX-B: optional torch.compile for ~15-25% throughput gain (PyTorch ≥ 2.0)
        if ROLE_CLASSIFICATION_CONFIG.get("use_torch_compile", False):
            try:
                self.model = torch.compile(self.model)
                logger.info("RoleClassifier: torch.compile() applied")
            except Exception as e:
                logger.warning(f"RoleClassifier: torch.compile() failed, continuing without: {e}")

    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        return_probabilities: bool = True
    ) -> Union[Dict, List[Dict]]:
        """
        Predict roles for text chunks.

        Returns dicts with keys: role, confidence, probabilities (optional).

        FIX-C: removed the reactive NCCL try/except fallback — that block was
        masking the DataParallel error for one or more batches before triggering,
        which could silently corrupt the predictions for those batches. The root
        cause (DataParallel) is now fixed at _load_model time.

        FIX-D: pin_memory() is only valid when the source tensor is on CPU and
        the destination is CUDA. Guard added so CPU-only machines don't error.
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        self.model.eval()
        predictions = []

        use_cuda = self.device.type == "cuda"
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self._use_amp and use_cuda
            else contextlib.nullcontext()
        )

        with torch.no_grad(), amp_ctx:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                encodings = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )

                # FIX-D: pin_memory() only works for CPU→CUDA transfers
                if use_cuda:
                    input_ids      = encodings["input_ids"].pin_memory().to(self.device, non_blocking=True)
                    attention_mask = encodings["attention_mask"].pin_memory().to(self.device, non_blocking=True)
                else:
                    input_ids      = encodings["input_ids"].to(self.device)
                    attention_mask = encodings["attention_mask"].to(self.device)

                # FIX-C: plain forward pass — no NCCL fallback needed
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                probs = torch.softmax(outputs.logits.float(), dim=1).cpu().numpy()
                pred_labels = np.argmax(probs, axis=1)

                for pred_label, prob_dist in zip(pred_labels, probs):
                    result = {
                        "role":       self.id2label[pred_label],
                        "confidence": float(prob_dist[pred_label]),
                    }
                    if return_probabilities:
                        result["probabilities"] = {
                            self.id2label[k]: float(v)
                            for k, v in enumerate(prob_dist)
                        }
                    predictions.append(result)

        return predictions[0] if single_input else predictions

    def classify_chunks(
        self,
        chunks: List[Dict],
        text_field: str = "text",
        batch_size: int = 32,
        add_to_chunks: bool = True,
        show_progress: bool = True
    ) -> List[Dict]:
        """Classify a list of chunk dicts. Adds 'role_prediction' field to each chunk."""
        texts = [chunk.get(text_field, "") for chunk in chunks]
        logger.info(f"Classifying {len(chunks)} chunks...")

        all_predictions = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Classifying chunks")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_predictions = self.predict(
                batch_texts,
                batch_size=len(batch_texts),
                return_probabilities=True
            )
            all_predictions.extend(batch_predictions)

        if add_to_chunks:
            for chunk, prediction in zip(chunks, all_predictions):
                chunk["role_prediction"] = prediction

        role_counts: Dict[str, int] = {}
        for pred in all_predictions:
            role = pred["role"]
            role_counts[role] = role_counts.get(role, 0) + 1

        logger.info("Role distribution: " + ", ".join(
            f"{role}: {count}" for role, count in sorted(role_counts.items())
        ))
        return chunks

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        output_dir: str = "./role_classifier_model",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        save_best_model: bool = True
    ):
        """Fine-tune the classifier on labeled data."""
        logger.info(f"Starting training with {len(train_texts)} examples")

        train_dataset = ChunkDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = None
        if val_texts and val_labels:
            val_dataset = ChunkDataset(val_texts, val_labels, self.tokenizer, self.max_length)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch" if save_best_model else "no",
            load_best_model_at_end=save_best_model and val_dataset is not None,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_total_limit=2,
            metric_for_best_model="accuracy" if val_dataset else None,
            greater_is_better=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics if val_dataset else None
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        role_config = {
            "role_definitions": self.role_definitions,
            "label2id": self.label2id,
            "id2label": {int(k): v for k, v in self.id2label.items()}
        }
        with open(Path(output_dir) / "role_config.json", "w") as f:
            json.dump(role_config, f, indent=2)

        logger.info(f"Training complete. Model saved to {output_dir}")

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": float((predictions == labels).mean())}

    def save(self, output_dir: str):
        """Save the model and configuration.

        FIX-E: unwrap DataParallel before saving so the checkpoint is always
        a plain model that can be loaded on any number of GPUs.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # FIX-E: strip any accidental DataParallel wrapper before saving
        model_to_save = (
            self.model.module
            if isinstance(self.model, torch.nn.DataParallel)
            else self.model
        )
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        role_config = {
            "role_definitions": self.role_definitions,
            "label2id": self.label2id,
            "id2label": {int(k): v for k, v in self.id2label.items()}
        }
        with open(Path(output_dir) / "role_config.json", "w") as f:
            json.dump(role_config, f, indent=2)

        logger.info(f"Model saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str, device: Optional[str] = None) -> "RoleClassifier":
        """Load a saved model from directory (reads label mapping from config.json)."""
        with open(Path(model_dir) / "config.json", "r") as f:
            model_config = json.load(f)

        id2label = {int(k): v for k, v in model_config["id2label"].items()}
        role_definitions = [id2label[i] for i in sorted(id2label.keys())]

        classifier = cls(
            role_definitions=role_definitions,
            model_name=model_dir,
            device=device
        )

        logger.info(f"Model loaded from {model_dir} with roles: {role_definitions}")
        return classifier


def create_classifier_from_config(config: Optional[Dict] = None) -> Optional[RoleClassifier]:
    """
    Create a RoleClassifier from config.py settings.

    With use_finetuned=True (default), loads the fine-tuned model from
    finetuned_model_path. Falls back to a base DistilBERT model if the
    path doesn't exist.
    """
    if config is None:
        try:
            from config import ROLE_CLASSIFICATION_CONFIG
            config = ROLE_CLASSIFICATION_CONFIG
        except ImportError:
            logger.error("Could not import ROLE_CLASSIFICATION_CONFIG from config.py")
            return None

    if not config.get("enabled", True):
        logger.info("Role classification is disabled in config")
        return None

    if config.get("use_finetuned", True):
        model_path = config.get("finetuned_model_path", "./final_model")
        if model_path and Path(model_path).exists():
            logger.info(f"Loading fine-tuned model from: {model_path}")
            return RoleClassifier.load(model_path, device=config.get("device"))
        else:
            logger.error(f"Fine-tuned model path not found: {model_path}")
            raise FileNotFoundError(f"Fine-tuned model not found at: {model_path}")

    # Fallback: base model (not recommended for production)
    model_name = config.get("model_name", "distilbert-base-uncased")
    role_definitions = config.get("role_definitions", [
        "Arguments", "Precedents", "Facts", "Issues",
        "Reasoning", "Decision", "Statute", "Preamble", "Others"
    ])
    logger.warning(f"Using base (untrained) model: {model_name}. Results will be poor.")
    return RoleClassifier(
        role_definitions=role_definitions,
        model_name=model_name,
        device=config.get("device"),
        max_length=config.get("max_length", 512)
    )


def prepare_training_data_from_annotated_chunks(
    annotated_chunks: List[Dict],
    role_field: str = "role",
    text_field: str = "text",
    role_definitions: Optional[List[str]] = None
) -> Tuple[List[str], List[int], List[str]]:
    """Prepare training data from annotated chunks."""
    texts = []
    roles = []

    for chunk in annotated_chunks:
        if text_field in chunk and role_field in chunk:
            texts.append(chunk[text_field])
            roles.append(chunk[role_field])

    if role_definitions is None:
        role_definitions = sorted(list(set(roles)))
        logger.info(f"Auto-detected {len(role_definitions)} roles: {role_definitions}")

    label2id = {role: idx for idx, role in enumerate(role_definitions)}
    labels = [label2id[role] for role in roles]

    return texts, labels, role_definitions