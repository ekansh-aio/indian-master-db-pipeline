"""
FUTURE/TODO: Fine-tuned Role Classifier for Legal Document Chunks using DistilBERT

⚠️ THIS MODULE IS NOT CURRENTLY IN USE ⚠️

This module provides fine-tuned classification of legal document chunks into custom roles.
Uses DistilBERT as the base model with a custom classification head.

STATUS: This classifier requires a fine-tuned model to work effectively. 
        Without fine-tuning, it produces low confidence scores and poor results.
        
CURRENT SOLUTION: Use role_classifier_embedding.py instead, which uses 
                  semantic similarity and requires no training.

FUTURE USE: Keep this module for when you have:
            1. Annotated training data for your specific roles
            2. Time and resources to fine-tune a model
            3. Need for higher accuracy than embedding-based approach

To use in future:
    1. Prepare annotated training data
    2. Fine-tune the model using the train() method
    3. Update config.py to use the fine-tuned model
    4. Replace role_classifier_embedding with this module in pipeline

UPDATED: Now reads configuration from config.py
"""
import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import json
from pathlib import Path
from tqdm import tqdm

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
    Legal document chunk role classifier using DistilBERT.
    
    Supports custom role definitions and can be trained or used for inference.
    """
    
    def __init__(
        self,
        role_definitions: List[str],
        model_name: str = "distilbert-base-uncased",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize the role classifier.
        
        Args:
            role_definitions: List of role names (e.g., ['facts', 'reasoning', 'conclusion'])
            model_name: HuggingFace model identifier or path to fine-tuned model
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum token length for text
        """
        self.role_definitions = role_definitions
        self.num_labels = len(role_definitions)
        self.max_length = max_length
        
        # Role mapping
        self.label2id = {role: idx for idx, role in enumerate(role_definitions)}
        self.id2label = {idx: role for idx, role in enumerate(role_definitions)}
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing RoleClassifier with {self.num_labels} roles")
        logger.info(f"Device: {self.device}")
        logger.info(f"Roles: {role_definitions}")
        
        # Load tokenizer and model
        self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load tokenizer and model."""
        try:
            # Try loading as a fine-tuned model first
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.role_definitions)
            )
            logger.info(f"Loaded fine-tuned model from: {model_name}")
        except Exception as e:
            # Load base model
            logger.warning(f"Could not load as fine-tuned model, loading base: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.role_definitions)
            )
            logger.info(f"Loaded base model: {model_name}")
        
        # Move model to the appropriate device (CPU/GPU)
        self.model.to(self.device) # type: ignore
        self.model.eval()
    
    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        return_probabilities: bool = True
    ) -> Union[Dict, List[Dict]]:
        """
        Predict roles for text chunks.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for inference
            return_probabilities: Whether to return probability scores
        
        Returns:
            Single prediction dict or list of prediction dicts with:
            - role: predicted role name
            - confidence: confidence score (0-1)
            - probabilities: dict of role -> probability (if return_probabilities=True)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                # Predict
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get probabilities
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                # Get predictions
                pred_labels = np.argmax(probs, axis=1)
                
                # Format results
                for j, (pred_label, prob_dist) in enumerate(zip(pred_labels, probs)):
                    result = {
                        'role': self.id2label[pred_label],
                        'confidence': float(prob_dist[pred_label])
                    }
                    
                    if return_probabilities:
                        result['probabilities'] = {
                            self.id2label[k]: float(v)
                            for k, v in enumerate(prob_dist)
                        }
                    
                    predictions.append(result)
        
        return predictions[0] if single_input else predictions
    
    def classify_chunks(
        self,
        chunks: List[Dict],
        text_field: str = 'text',
        batch_size: int = 32,
        add_to_chunks: bool = True,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Classify a list of chunk dictionaries.
        
        Args:
            chunks: List of chunk dicts
            text_field: Field name containing the text
            batch_size: Batch size for inference
            add_to_chunks: Whether to add predictions to chunk dicts (modifies in-place)
            show_progress: Show progress bar
        
        Returns:
            List of chunks with added 'role_prediction' field
        """
        texts = [chunk.get(text_field, '') for chunk in chunks]
        
        logger.info(f"Classifying {len(chunks)} chunks...")
        
        # Batch prediction with progress bar
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
        
        # Add predictions to chunks
        if add_to_chunks:
            for chunk, prediction in zip(chunks, all_predictions):
                chunk['role_prediction'] = prediction
        
        logger.info(f"Classification complete")
        
        # Log role distribution
        role_counts = {}
        for pred in all_predictions:
            role = pred['role']
            role_counts[role] = role_counts.get(role, 0) + 1
        
        logger.info("Role distribution:")
        for role, count in sorted(role_counts.items()):
            percentage = (count / len(all_predictions)) * 100
            logger.info(f"  {role}: {count} ({percentage:.1f}%)")
        
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
        """
        Fine-tune the classifier on labeled data.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels (as integer indices)
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            output_dir: Directory to save the model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_best_model: Save the best model based on validation
        """
        logger.info(f"Starting training with {len(train_texts)} examples")
        
        # Create datasets
        train_dataset = ChunkDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            self.max_length
        )
        
        val_dataset = None
        if val_texts and val_labels:
            val_dataset = ChunkDataset(
                val_texts,
                val_labels,
                self.tokenizer,
                self.max_length
            )
            logger.info(f"Validation set: {len(val_texts)} examples")
        
        # Training arguments
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
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics if val_dataset else None
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save role definitions
        role_config = {
            'role_definitions': self.role_definitions,
            'label2id': self.label2id,
            'id2label': {int(k): v for k, v in self.id2label.items()}
        }
        
        with open(Path(output_dir) / 'role_config.json', 'w') as f:
            json.dump(role_config, f, indent=2)
        
        logger.info("Training complete!")
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = (predictions == labels).mean()
        
        return {'accuracy': accuracy}
    
    def save(self, output_dir: str):
        """Save the model and configuration."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save role definitions
        role_config = {
            'role_definitions': self.role_definitions,
            'label2id': self.label2id,
            'id2label': {int(k): v for k, v in self.id2label.items()}
        }
        
        with open(Path(output_dir) / 'role_config.json', 'w') as f:
            json.dump(role_config, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir: str, device: Optional[str] = None):
        """
        Load a saved model.
        
        Args:
            model_dir: Directory containing the saved model
            device: Device to load on
        
        Returns:
            RoleClassifier instance
        """
        # Read label mappings directly from the model's config.json
        with open(Path(model_dir) / 'config.json', 'r') as f:
            model_config = json.load(f)

        id2label = {int(k): v for k, v in model_config['id2label'].items()}
        role_definitions = [id2label[i] for i in sorted(id2label.keys())]

        # Create classifier
        classifier = cls(
            role_definitions=role_definitions,
            model_name=model_dir,
            device=device
        )
        
        logger.info(f"Model loaded from {model_dir}")
        return classifier


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_default_legal_classifier(device: Optional[str] = None) -> RoleClassifier:
    """
    Create a classifier with common legal document roles.
    
    Default roles:
    - case_citation: Citations and references to other cases
    - facts: Factual background and circumstances
    - procedural_history: Court proceedings and timeline
    - issue: Legal questions presented
    - reasoning: Legal analysis and arguments
    - holding: Court's decision and ruling
    - disposition: Final order or judgment
    - dissent: Dissenting opinions
    - other: Other content
    """
    default_roles = [
        'case_citation',
        'facts',
        'procedural_history',
        'issue',
        'reasoning',
        'holding',
        'disposition',
        'dissent',
        'other'
    ]
    
    return RoleClassifier(role_definitions=default_roles, device=device)


def create_classifier_from_config(config: Optional[Dict] = None) -> RoleClassifier:
    """
    Create a classifier using settings from config.py.
    
    Args:
        config: Configuration dictionary. If None, will import from config.py
    
    Returns:
        RoleClassifier instance configured from config
    """
    if config is None:
        # Import config from config.py
        try:
            from config import ROLE_CLASSIFICATION_CONFIG
            config = ROLE_CLASSIFICATION_CONFIG
        except ImportError:
            logger.warning("Could not import config.py, using defaults")
            return create_default_legal_classifier()
    
    # Check if role classification is enabled
    if not config.get('enabled', True):
        logger.warning("Role classification is disabled in config")
        return None # type: ignore
    
    # Determine which model to use
    if config.get('use_finetuned', False):
        model_path = config.get('finetuned_model_path')
        if model_path and Path(model_path).exists():
            logger.info(f"Loading fine-tuned model from: {model_path}")
            return RoleClassifier.load(model_path, device=config.get('device'))
        else:
            logger.warning(f"Fine-tuned model path not found: {model_path}, using base model")
    
    # Use base model
    model_name = config.get('model_name', 'distilbert-base-uncased')
    role_definitions = config.get('role_definitions', [
        'metadata', 'procedural_history', 'factual_background',
        'legal_analysis', 'application', 'orders', 'other'
    ])
    max_length = config.get('max_length', 512)
    device = config.get('device', None)
    
    logger.info(f"Creating classifier with model: {model_name}")
    logger.info(f"Role definitions: {role_definitions}")
    
    return RoleClassifier(
        role_definitions=role_definitions,
        model_name=model_name,
        device=device,
        max_length=max_length
    )


def prepare_training_data_from_annotated_chunks(
    annotated_chunks: List[Dict],
    role_field: str = 'role',
    text_field: str = 'text',
    role_definitions: Optional[List[str]] = None
) -> Tuple[List[str], List[int], List[str]]:
    """
    Prepare training data from annotated chunks.
    
    Args:
        annotated_chunks: List of chunk dicts with role annotations
        role_field: Field name containing the role label
        text_field: Field name containing the text
        role_definitions: List of valid roles (auto-detected if None)
    
    Returns:
        (texts, labels, role_definitions)
    """
    texts = []
    roles = []
    
    for chunk in annotated_chunks:
        if text_field in chunk and role_field in chunk:
            texts.append(chunk[text_field])
            roles.append(chunk[role_field])
    
    # Auto-detect roles if not provided
    if role_definitions is None:
        role_definitions = sorted(list(set(roles)))
        logger.info(f"Auto-detected {len(role_definitions)} roles: {role_definitions}")
    
    # Convert roles to indices
    label2id = {role: idx for idx, role in enumerate(role_definitions)}
    labels = [label2id[role] for role in roles]
    
    return texts, labels, role_definitions