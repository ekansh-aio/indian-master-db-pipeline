"""
Configuration for the legal document processing pipeline.
Credentials are loaded from .env.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ========================================
# LOGGING
# ========================================
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "log_file": os.getenv("LOG_FILE", "pipeline.log"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# ========================================
# AZURE DATA LAKE STORAGE
# ========================================
ADLS_CONFIG = {
    "account_name":   os.getenv("ADLS_ACCOUNT_NAME"),
    "account_key":    os.getenv("ADLS_ACCOUNT_KEY"),
    "container_name": os.getenv("ADLS_CONTAINER_NAME"),
    "file_pattern":   "*.json",
    "recursive":      True
}

# ========================================
# ELASTICSEARCH
# ========================================
ES_CONFIG = {
    "url":       os.getenv("ES_URL"),
    "api_key":   os.getenv("ES_API_KEY"),
    "user":      os.getenv("ES_USER"),
    "password":  os.getenv("ES_PASS"),
}

# ========================================
# DOCUMENT TYPE CONFIG
# doc_type 0 = High Court, 1 = Supreme Court
# ========================================
DOC_TYPE_CONFIG = {
    0: {
        "name":            "High Court",
        "jurisdiction":    "India",
        "adls_input_path": os.getenv("HC_INPUT_PATH", "app/High_Court_Judgements/"),
    },
    1: {
        "name":            "Supreme Court",
        "jurisdiction":    "India",
        "adls_input_path": os.getenv("SC_INPUT_PATH", "app/Supreme_Court_Judgements/"),
    },
}

# ========================================
# ROLE WEIGHTS (for top-K selection during ES upload)
# ========================================
ROLE_WEIGHTS = {
    "Decision":   3.0,
    "Precedents": 3.0,
    "Issues":     2.5,
    "Preamble":   2.5,
    "Facts":      2.5,
    "Statute":    1.5,
    "Reasoning":  0.4,
    "Arguments":  0.3,
    "Others":     0.2,
}

# ========================================
# EMBEDDING MODEL
# ========================================
EMBEDDING_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    "dimensions": int(os.getenv("EMBEDDING_DIMENSIONS", "384")),  # all-MiniLM-L6-v2 = 384-dim
    "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "1024")),
    "multi_gpu":  os.getenv("EMBEDDING_MULTI_GPU", "true").lower() == "true",
    "use_amp":    os.getenv("EMBEDDING_USE_AMP", "true").lower() == "true",
}

# ========================================
# ROLE CLASSIFICATION
# ========================================
ROLE_CLASSIFICATION_CONFIG = {
    "enabled":              os.getenv("ROLE_CLASSIFICATION_ENABLED", "true").lower() == "true",
    "use_finetuned":        os.getenv("USE_FINETUNED_ROLE_MODEL", "true").lower() == "true",
    "finetuned_model_path": os.getenv("FINETUNED_ROLE_MODEL_PATH", "./final_model"),
    "device":               os.getenv("ROLE_DEVICE", None),
    "batch_size":           int(os.getenv("ROLE_CLASSIFICATION_BATCH_SIZE", "256")),
    "max_length":           int(os.getenv("ROLE_MAX_LENGTH", "512")),
    "use_amp":              os.getenv("ROLE_USE_AMP", "true").lower() == "true",
    "num_workers":          int(os.getenv("ROLE_NUM_WORKERS", "4")),
    "num_gpus":             int(os.getenv("ROLE_NUM_GPUS", "6")),
}

# ========================================
# SEMANTIC CHUNKING
# ========================================
CHUNKING_CONFIG = {
    "similarity_threshold":    float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
    "min_sentences_per_chunk": int(os.getenv("MIN_SENTENCES_PER_CHUNK", "3")),
    "max_sentences_per_chunk": int(os.getenv("MAX_SENTENCES_PER_CHUNK", "10")),
    "min_chunk_size":          int(os.getenv("MIN_CHUNK_SIZE", "100")),
    "compute_doc_similarity":  os.getenv("COMPUTE_DOC_SIMILARITY", "true").lower() == "true",
    "top_k":                   int(os.getenv("TOP_K_CHUNKS")) if os.getenv("TOP_K_CHUNKS") else None,
    "top_k_method":            os.getenv("TOP_K_METHOD", "doc_similarity"),
    "num_gpus":                int(os.getenv("CHUNKING_NUM_GPUS", "6")),
    "device":                  os.getenv("CHUNKING_DEVICE", None),
}

# ========================================
# PROCESSING
# ========================================
PROCESSING_CONFIG = {
    "batch_size": int(os.getenv("BATCH_SIZE", "10")),
    "skip_errors": os.getenv("SKIP_ERRORS", "true").lower() == "true"
}

# ========================================
# PIPELINE
# ========================================
PIPELINE_CONFIG = {
    "max_documents":         int(os.getenv("MAX_DOCUMENTS")) if os.getenv("MAX_DOCUMENTS") else None,
    "processing_batch_size": int(os.getenv("PROCESSING_BATCH_SIZE", "64")),
    "io_workers":            int(os.getenv("IO_WORKERS", "64")),
}

# ========================================
# VALIDATION
# ========================================
def validate_config():
    errors = []

    if not ADLS_CONFIG["account_name"]:
        errors.append("ADLS_ACCOUNT_NAME not set")
    if not ADLS_CONFIG["account_key"]:
        errors.append("ADLS_ACCOUNT_KEY not set")
    if not ADLS_CONFIG["container_name"]:
        errors.append("ADLS_CONTAINER_NAME not set")

    if ROLE_CLASSIFICATION_CONFIG["enabled"] and ROLE_CLASSIFICATION_CONFIG["use_finetuned"]:
        model_path = ROLE_CLASSIFICATION_CONFIG["finetuned_model_path"]
        if not model_path or not os.path.exists(model_path):
            errors.append(f"Fine-tuned model path does not exist: {model_path}")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True
