"""
Enhanced configuration file for the legal document processing pipeline.
All settings can be overridden by environment variables.

UPDATED: Added role classification configuration
"""
import os
from dotenv import load_dotenv
from role_desc import ROLE_DESCRIPTIONS_DICT


# Load environment variables
load_dotenv()

# ========================================
# LOGGING CONFIGURATION
# ========================================
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),  # DEBUG, INFO, WARNING, ERROR
    "log_file": os.getenv("LOG_FILE", "pipeline.log"),  # Set to None for console only
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# ========================================
# AZURE DATA LAKE STORAGE (ADLS) CONFIGURATION
# ========================================
ADLS_CONFIG = {
    "account_name": os.getenv("ADLS_ACCOUNT_NAME"),
    "account_key": os.getenv("ADLS_ACCOUNT_KEY"),
    "container_name": os.getenv("ADLS_CONTAINER_NAME"),
    "input_path": os.getenv("ADLS_INPUT_PATH", "raw/newapp"),  # Base path in ADLS
    "file_pattern": "*.json",  # Pattern to match files
    "recursive": True  # Whether to search recursively
}

# ========================================
# AZURE AI SEARCH CONFIGURATION
# ========================================
SEARCH_CONFIG = {
    "endpoint": os.getenv("SEARCH_ENDPOINT"),
    "key": os.getenv("SEARCH_KEY"),
    "index_name": os.getenv("INDEX_NAME", "legal-documents-index"),
    "upload_batch_size": int(os.getenv("SEARCH_UPLOAD_BATCH_SIZE", "100")),  # Batch size for uploading
    "max_retries": int(os.getenv("SEARCH_MAX_RETRIES", "3")),
    "retry_delay": float(os.getenv("SEARCH_RETRY_DELAY", "2.0"))
}

# ========================================
# EMBEDDING MODEL CONFIGURATION
# ========================================
EMBEDDING_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    "dimensions": int(os.getenv("EMBEDDING_DIMENSIONS", "768")),  # Must match model output
    "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))  # For encoding in batches
}

# ========================================
# ROLE CLASSIFICATION CONFIGURATION
# ========================================
ROLE_CLASSIFICATION_CONFIG = {
    # Enable/disable role classification
    "enabled": os.getenv("ROLE_CLASSIFICATION_ENABLED", "true").lower() == "true",
    
    # Classification method: "embedding" or "finetuned"
    # "embedding" uses semantic similarity (no training needed, works out of the box)
    # "finetuned" uses a fine-tuned transformer model (requires training data)
    "method": os.getenv("ROLE_CLASSIFICATION_METHOD", "embedding"),  # embedding | finetuned
    
    # ===== EMBEDDING-BASED CLASSIFICATION SETTINGS =====
    # Uses the same embedding model as the main pipeline
    # Compares chunk embeddings with role description embeddings
    
    # Confidence threshold for embedding-based classification (0.0-1.0)
    # Lower = more chunks classified, higher = stricter classification
    # Recommended: 0.25-0.4 for legal documents
    "confidence_threshold": float(os.getenv("ROLE_CONFIDENCE_THRESHOLD", "0.6")),
    
    # How to aggregate multiple descriptions per role: "max" or "mean"
    # "max" = use highest similarity among descriptions (recommended)
    # "mean" = use average similarity among descriptions
    "aggregation_method": os.getenv("ROLE_AGGREGATION_METHOD", "mean"),
    
    # Role descriptions dictionary for embedding-based classification
    # Maps role names to lists of description sentences
    # These descriptions are embedded and compared with chunk embeddings
    # Customize these based on your document types and needs!
    "role_descriptions_dict": ROLE_DESCRIPTIONS_DICT,
    
    # ===== FINE-TUNED MODEL SETTINGS (FUTURE USE) =====
    # Only used if method="finetuned"
    # Requires training data and fine-tuned model
    
    # Model configuration for fine-tuned approach
    "model_name": os.getenv("ROLE_MODEL_NAME", "microsoft/deberta-v3-base"),  # Base model
    "use_finetuned": os.getenv("USE_FINETUNED_ROLE_MODEL", "false").lower() == "true",
    "finetuned_model_path": os.getenv("FINETUNED_ROLE_MODEL_PATH", "./models/role_classifier"),
    
    # Role definitions for fine-tuned approach (simple list of role names)
    # Only used if role_descriptions_dict is not provided
    "role_definitions": [
        role.strip() 
        for role in os.getenv(
            "ROLE_DEFINITIONS",
            "case_metadata , procedural_history , factual_background , legal_issues , legal_analysis , holding_and_conlusions ,others"
        ).split(',')
    ],
    
    # Processing settings (shared by both methods)
    "batch_size": int(os.getenv("ROLE_CLASSIFICATION_BATCH_SIZE", "32")),
    "max_length": int(os.getenv("ROLE_MAX_LENGTH", "512")),
    "device": os.getenv("ROLE_DEVICE", None),  # None for auto-detect, 'cuda' or 'cpu'
    
    # Output settings
    "add_probabilities": os.getenv("ROLE_ADD_PROBABILITIES", "true").lower() == "true",
}

# ========================================
# SEMANTIC CHUNKING CONFIGURATION
# ========================================
CHUNKING_CONFIG = {
    "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
    "min_sentences_per_chunk": int(os.getenv("MIN_SENTENCES_PER_CHUNK", "3")),
    "max_sentences_per_chunk": int(os.getenv("MAX_SENTENCES_PER_CHUNK", "10")),
    "min_chunk_size": int(os.getenv("MIN_CHUNK_SIZE", "100")),  # Minimum character count
    "compute_doc_similarity": os.getenv("COMPUTE_DOC_SIMILARITY", "true").lower() == "true",
    "top_k": int(os.getenv("TOP_K_CHUNKS", "4")) if os.getenv("TOP_K_CHUNKS") else None,  # None = all chunks
    "top_k_method": os.getenv("TOP_K_METHOD", "doc_similarity")  # or 'avg_similarity'
}

# ========================================
# PROCESSING CONFIGURATION
# ========================================
PROCESSING_CONFIG = {
    "batch_size": int(os.getenv("BATCH_SIZE", "10")),  # Documents per batch
    "num_workers": int(os.getenv("NUM_WORKERS", "4")),  # Parallel workers
    "use_parallel": os.getenv("USE_PARALLEL", "true").lower() == "true",
    "use_caching": os.getenv("USE_CACHING", "true").lower() == "true",
    "skip_errors": os.getenv("SKIP_ERRORS", "true").lower() == "true",
    "checkpoint_interval": int(os.getenv("CHECKPOINT_INTERVAL", "100"))  # Save progress every N docs
}

# ========================================
# TEXT CLEANING CONFIGURATION
# ========================================
CLEANING_CONFIG = {
    "remove_law_reports": True,
    "remove_judge_names": True,
    "remove_citations": True,
    "remove_page_headers": True,
    "normalize_whitespace": True
}

# ========================================
# PARAGRAPH SPLITTER CONFIGURATION (Legacy)
# ========================================
PARAGRAPH_SPLITTER_CONFIG = {
    "strategy": "double_newline",
    "custom_pattern": None,
    "fixed_size": 1000,
    "min_chunk_size": 50
}

# ========================================
# OUTPUT CONFIGURATION
# ========================================
OUTPUT_CONFIG = {
    "intermediate_dir": os.getenv("INTERMEDIATE_DIR", "output/intermediate"),
    "final_dir": os.getenv("FINAL_DIR", "output/final"),
    "save_intermediate": os.getenv("SAVE_INTERMEDIATE", "true").lower() == "true",
    "all_chunks_filename": "all_chunks.json",
    "top_k_chunks_filename": "top_k_chunks.json",
    "embeddings_filename": "chunks_with_embeddings.json",
    "stats_filename": "processing_stats.json"
}

# ========================================
# PIPELINE CONFIGURATION
# ========================================
PIPELINE_CONFIG = {
    "mode": os.getenv("PIPELINE_MODE", "full"),  # full, test, partial
    "max_documents": int(os.getenv("MAX_DOCUMENTS", "100")) if os.getenv("MAX_DOCUMENTS") else None,  # Limit for testing
    "create_index": os.getenv("CREATE_INDEX", "true").lower() == "true",  # Whether to create/update index
    "upload_to_search": os.getenv("UPLOAD_TO_SEARCH", "true").lower() == "true",  # Whether to upload results
}

# ========================================
# VALIDATION
# ========================================
def validate_config():
    """Validate that all required configuration values are present."""
    errors = []
    
    # Check ADLS config
    if not ADLS_CONFIG["account_name"]:
        errors.append("ADLS_ACCOUNT_NAME not set")
    if not ADLS_CONFIG["account_key"]:
        errors.append("ADLS_ACCOUNT_KEY not set")
    if not ADLS_CONFIG["container_name"]:
        errors.append("ADLS_CONTAINER_NAME not set")
    
    # Check Search config
    if PIPELINE_CONFIG["upload_to_search"]:
        if not SEARCH_CONFIG["endpoint"]:
            errors.append("SEARCH_ENDPOINT not set")
        if not SEARCH_CONFIG["key"]:
            errors.append("SEARCH_KEY not set")
    
    # Check role classification config
    if ROLE_CLASSIFICATION_CONFIG["enabled"]:
        if not ROLE_CLASSIFICATION_CONFIG["role_definitions"]:
            errors.append("ROLE_DEFINITIONS not set or empty")
        
        if ROLE_CLASSIFICATION_CONFIG["use_finetuned"]:
            model_path = ROLE_CLASSIFICATION_CONFIG["finetuned_model_path"]
            if not model_path or not os.path.exists(model_path):
                errors.append(f"Fine-tuned model path does not exist: {model_path}")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

# ========================================
# HELPER FUNCTIONS FOR BACKWARDS COMPATIBILITY
# ========================================
def load_processing_config():
    """Load processing configuration (for backward compatibility)."""
    return PROCESSING_CONFIG

def load_paragraph_splitter_config():
    """Load paragraph splitter configuration (for backward compatibility)."""
    return PARAGRAPH_SPLITTER_CONFIG

# Custom Legal Abbreviations (for backward compatibility)
CUSTOM_ABBREVIATIONS = [
    "Pvt.", "Pty.", "Intl.", "Dept.", "Assn.", "Bros.", "Mfg.", "Dist."
]

# Legacy file paths (kept for backward compatibility)
INPUT_FILE = "legislation_input.json"
OUTPUT_FILE = "vector_ready_legislation.json"
LOG_FILE = "processing.log"