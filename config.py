"""
Configuration for the legal document processing pipeline.
Credentials are loaded from .env. Paths and index names are defined here.
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
# AZURE DATA LAKE STORAGE — credentials only
# Paths are in DOC_TYPE_CONFIG below
# ========================================
ADLS_CONFIG = {
    "account_name":   os.getenv("ADLS_ACCOUNT_NAME"),
    "account_key":    os.getenv("ADLS_ACCOUNT_KEY"),
    "container_name": os.getenv("ADLS_CONTAINER_NAME"),
    "file_pattern":   "*.json",
    "recursive":      True
}

# ========================================
# AZURE AI SEARCH — credentials only
# Index names are in DOC_TYPE_CONFIG below
# ========================================
SEARCH_CONFIG = {
    "endpoint":          os.getenv("SEARCH_ENDPOINT"),
    "key":               os.getenv("SEARCH_KEY"),
    "upload_batch_size": int(os.getenv("SEARCH_UPLOAD_BATCH_SIZE", "100")),
    "max_retries":       int(os.getenv("SEARCH_MAX_RETRIES", "3")),
    "retry_delay":       float(os.getenv("SEARCH_RETRY_DELAY", "2.0"))
}

# ========================================
# DOCUMENT TYPE CONFIG
# doc_type 0 = High Court, 1 = Supreme Court
# Defines: ADLS input path, index names per index_type, jurisdiction
# ========================================
DOC_TYPE_CONFIG = {
    0: {
        "name":            "High Court",
        "jurisdiction":    "India",
        "adls_input_path": os.getenv("HC_INPUT_PATH", "app/High_Court_Judgements/"),
        "index_names": {
            0: "hc-ai-assistant",
            1: "hc-precedent-finder"
        }
    },
    1: {
        "name":            "Supreme Court",
        "jurisdiction":    "India",
        "adls_input_path": os.getenv("SC_INPUT_PATH", "app/Supreme_Court_Judgements/"),
        "index_names": {
            0: "sc-ai-assistant",
            1: "sc-precedent-finder"
        }
    },
}

# ========================================
# ROLE WEIGHTS
# Tunable here — no need to touch pipeline code
# ========================================

# AI Assistant: uniform weights — selection is purely proportional to
# the role distribution naturally present in each document
AI_ASSISTANT_ROLE_WEIGHTS = {
    "Arguments":  1.0,
    "Precedents": 1.0,
    "Facts":      1.0,
    "Issues":     1.0,
    "Reasoning":  1.0,
    "Decision":   1.0,
    "Statute":    1.0,
    "Preamble":   1.0,
    "Others":     1.0
}

# Precedent Finder: biased toward roles that carry legal significance
# for case-to-case precedent retrieval
PRECEDENT_FINDER_ROLE_WEIGHTS = {
    "Decision":   3.0,   # The actual ruling — most critical for precedent
    "Precedents": 3.0,   # Prior case citations — core of precedent finding
    "Issues":     2.0,   # Legal questions framed by the court
    "Reasoning":  2.0,   # Court's legal analysis supporting the ruling
    "Arguments":  1.0,   # Parties' submissions — some relevance
    "Facts":      0.5,   # Case-specific background — low precedent value
    "Statute":    0.5,   # Statutory text — relevant but not precedent-defining
    "Preamble":   0.3,   # Introductory formalities — minimal value
    "Others":     0.2    # Noise
}

# ========================================
# INDEX TYPE CONFIG
# index_type 0 = AI Assistant, 1 = Precedent Finder
# ========================================
INDEX_TYPE_CONFIG = {
    0: {
        "name": "AI Assistant",
        "role_weights": AI_ASSISTANT_ROLE_WEIGHTS
    },
    1: {
        "name": "Precedent Finder",
        "role_weights": PRECEDENT_FINDER_ROLE_WEIGHTS
    }
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
    "create_index":          os.getenv("CREATE_INDEX", "true").lower() == "true",
    "upload_to_search":      os.getenv("UPLOAD_TO_SEARCH", "true").lower() == "true",
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

    if PIPELINE_CONFIG["upload_to_search"]:
        if not SEARCH_CONFIG["endpoint"]:
            errors.append("SEARCH_ENDPOINT not set")
        if not SEARCH_CONFIG["key"]:
            errors.append("SEARCH_KEY not set")

    if ROLE_CLASSIFICATION_CONFIG["enabled"] and ROLE_CLASSIFICATION_CONFIG["use_finetuned"]:
        model_path = ROLE_CLASSIFICATION_CONFIG["finetuned_model_path"]
        if not model_path or not os.path.exists(model_path):
            errors.append(f"Fine-tuned model path does not exist: {model_path}")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True
