"""
Semantic Pipeline Processor Module (FIXED VERSION)
- Generates unique document IDs from source file paths
- Removes unnecessary metadata (similarity scores, char positions, num_sentences)
- Adds traceability (original_source_path, all_chunks_path)
- Properly structures output to mirror input directory structure
"""
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils.config_loader import load_processing_config
from core.legal_text_cleaner import clean_legal_text, LegalTextCleaner
from core.semantic_chunker import split_into_semantic_chunks, SemanticChunker

logger = logging.getLogger(__name__)


def generate_document_id_from_path(source_file_path: str) -> str:
    """
    Generate a unique document ID from the source file path.
    
    Examples:
        'raw/newapp/decisions/queensland/2005/doc.json' -> 'decisions_queensland_2005_doc'
        'raw/newapp/legislation/commonwealth/1906/act.json' -> 'legislation_commonwealth_1906_act'
    
    Args:
        source_file_path: Path to source file in ADLS
        
    Returns:
        Unique document ID string
    """
    # Remove file extension
    path_without_ext = source_file_path.replace('.json', '')
    
    # Split path into components
    parts = Path(path_without_ext).parts
    
    # Skip generic prefixes like 'raw', 'newapp', etc.
    skip_prefixes = {'raw', 'newapp', 'input', 'data'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    
    # Join with underscores to create ID
    doc_id = '_'.join(relevant_parts)
    
    # Clean up the ID (remove special characters, normalize)
    doc_id = doc_id.replace('-', '_').replace(' ', '_').lower()
    
    # If ID is too long, create a hash-based ID
    if len(doc_id) > 200:
        # Use last meaningful part + hash
        base = relevant_parts[-1] if relevant_parts else 'doc'
        hash_suffix = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        doc_id = f"{base}_{hash_suffix}"
    
    return doc_id


def get_all_chunks_adls_path(source_file_path: str, base_output_path: str = "processed") -> str:
    """
    Generate the ADLS path for all_chunks.json that mirrors the input structure.
    
    Example:
        Input:  'raw/newapp/decisions/queensland/2005/document.json'
        Output: 'processed/decisions/queensland/2005/document_all_chunks.json'
    
    Args:
        source_file_path: Original source file path
        base_output_path: Base output directory (default: 'processed')
        
    Returns:
        ADLS path for all_chunks.json
    """
    # Remove file extension
    path_without_ext = source_file_path.replace('.json', '')
    
    # Split path
    parts = Path(path_without_ext).parts
    
    # Skip generic prefixes
    skip_prefixes = {'raw', 'newapp', 'input', 'data'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    
    # Reconstruct path under processed directory
    if relevant_parts:
        # Join all parts except the last one (filename) to create directory structure
        directory_parts = relevant_parts[:-1]
        filename = relevant_parts[-1]
        
        if directory_parts:
            output_dir = '/'.join(directory_parts)
            return f"{base_output_path}/{output_dir}/{filename}_all_chunks.json"
        else:
            return f"{base_output_path}/{filename}_all_chunks.json"
    else:
        # Fallback if no relevant parts found
        return f"{base_output_path}/unknown_all_chunks.json"


@dataclass
class DocumentMetadata:
    """Structured document metadata."""
    raw_metadata: Dict
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentMetadata':
        """
        Create from dictionary, preserving all metadata fields except internal ones.
        """
        # Extract metadata, excluding text and internal fields
        excluded_fields = {'text', 'embedding', '_source_file'}
        metadata = {k: v for k, v in data.items() if k not in excluded_fields}
        return cls(raw_metadata=metadata)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self.raw_metadata.copy()


@dataclass
class SemanticProcessingStats:
    """Statistics for semantic pipeline processing."""
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    total_top_k_chunks: int = 0
    avg_chunks_per_doc: float = 0.0
    total_time: float = 0.0
    avg_time_per_doc: float = 0.0
    failed_doc_ids: List[str] = []
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    
    def __post_init__(self):
        if self.failed_doc_ids is None:
            self.failed_doc_ids = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class BatchSemanticDocumentProcessor:
    """
    Document processor using semantic chunking with improved ID generation and metadata.
    """
    
    def __init__(
        self,
        use_caching: bool = True,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.8,
        min_sentences_per_chunk: int = 2,
        max_sentences_per_chunk: int = 10,
        top_k: Optional[int] = None,
        top_k_method: str = "doc_similarity",
        batch_size: Optional[int] = 10,
        num_workers: int = 4,
        base_output_path: str = "processed"
    ):
        """
        Initialize semantic document processor.
        
        Args:
            use_caching: Whether to cache instances
            model_name: Sentence transformer model name
            similarity_threshold: Similarity threshold for chunking
            min_sentences_per_chunk: Minimum sentences per chunk
            max_sentences_per_chunk: Maximum sentences per chunk
            top_k: Number of top chunks to select (None = all chunks)
            top_k_method: Method to select top k ('doc_similarity' or 'avg_similarity')
            batch_size: Number of documents per batch
            num_workers: Number of parallel workers
            base_output_path: Base path for output in ADLS (default: 'processed')
        """
        logger.info("Initializing BatchSemanticDocumentProcessor (FIXED)")

        processing_cfg = load_processing_config()

        self.batch_size = batch_size or processing_cfg.get("batch_size", 10)
        self.num_workers = num_workers
        self.use_caching = use_caching
        self.top_k = top_k
        self.top_k_method = top_k_method
        self.base_output_path = base_output_path
        
        # Semantic chunker parameters
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk

        if use_caching:
            self.cleaner = LegalTextCleaner()
            self.semantic_chunker = SemanticChunker(
                model_name=model_name,
                similarity_threshold=similarity_threshold,
                min_sentences_per_chunk=min_sentences_per_chunk,
                max_sentences_per_chunk=max_sentences_per_chunk
            )

            logger.info(
                f"Processor initialized: model={model_name}, "
                f"threshold={similarity_threshold}, top_k={top_k}, "
                f"batch_size={self.batch_size}, workers={num_workers}"
            )
        else:
            self.cleaner = None
            self.semantic_chunker = None
    
    def process_document(self, doc: Dict) -> Optional[Tuple[List[Dict], List[Dict], str, str]]:
        """
        Process a single legal document.
        
        Args:
            doc: Document dictionary with 'text' and metadata
            
        Returns:
            Tuple of (all_chunks, top_k_chunks, doc_id, all_chunks_path) or None on failure
        """
        # Get source file path (added by ADLSFetcher)
        source_file_path = doc.get("_source_file", "unknown_source.json")
        
        # Generate unique document ID from source path
        doc_id = generate_document_id_from_path(source_file_path)
        
        # Generate all_chunks ADLS path
        all_chunks_path = get_all_chunks_adls_path(source_file_path, self.base_output_path)
        
        try:
            # Extract metadata (excluding text and internal fields)
            metadata = DocumentMetadata.from_dict(doc)
            
            # Step 1: Clean text
            start_time = time.time()
            
            if self.use_caching and self.cleaner:
                cleaned = self.cleaner.clean(doc["text"])
            else:
                cleaned = clean_legal_text(doc["text"])
            
            clean_time = time.time() - start_time
            
            if not cleaned:
                logger.warning(f"Document {doc_id} resulted in empty text after cleaning")
                return None
            
            # Step 2: Semantic chunking with document similarity
            start_time = time.time()
            
            if self.use_caching and self.semantic_chunker:
                chunks, doc_embedding = self.semantic_chunker.split(
                    cleaned, 
                    compute_doc_similarity=True
                )
            else:
                chunks, doc_embedding = split_into_semantic_chunks(
                    cleaned,
                    model_name=self.model_name,
                    similarity_threshold=self.similarity_threshold,
                    min_sentences_per_chunk=self.min_sentences_per_chunk,
                    max_sentences_per_chunk=self.max_sentences_per_chunk,
                    compute_doc_similarity=True
                )
            
            chunk_time = time.time() - start_time
            
            if not chunks:
                logger.warning(f"Document {doc_id} resulted in no chunks")
                return None
            
            # Step 3: Create final chunks with cleaned metadata
            final_chunks_all = []
            final_chunks_topk = []
            metadata_dict = metadata.to_dict()
            
            for idx, chunk in enumerate(chunks):
                # Create chunk for ALL chunks (stored in ADLS)
                # This includes traceability but excludes unnecessary metadata
                chunk_all = {
                    "id": f"{doc_id}_{idx}",
                    "chunk_id": f"{doc_id}_{idx}",
                    "doc_id": doc_id,
                    "original_source_path": source_file_path,  # Traceability to original
                    **metadata_dict,  # Original document metadata
                    "text": chunk["text"]
                    # NOTE: Removed start_char, end_char, num_sentences, avg_similarity
                }
                final_chunks_all.append(chunk_all)
                
                # Store chunk with doc_similarity for top-k selection
                chunk["chunk_id"] = f"{doc_id}_{idx}"
                chunk["doc_id"] = doc_id
            
            # Step 4: Select top-k chunks if configured
            if self.top_k is not None and self.top_k < len(chunks):
                # Sort by the configured method
                sort_key = self.top_k_method  # 'doc_similarity' or 'avg_similarity'
                sorted_chunks = sorted(
                    chunks,
                    key=lambda x: x.get(sort_key, 0.0),
                    reverse=True
                )
                selected_chunks = sorted_chunks[:self.top_k]
            else:
                selected_chunks = chunks
            
            # Step 5: Create final chunks for TOP-K (sent to search)
            # These need both paths for traceability
            for chunk in selected_chunks:
                chunk_idx = int(chunk["chunk_id"].split("_")[-1])
                chunk_topk = {
                    "id": chunk["chunk_id"],
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": doc_id,
                    "original_source_path": source_file_path,  # Path to original document
                    "all_chunks_path": all_chunks_path,  # Path to all_chunks.json in ADLS
                    **metadata_dict,
                    "text": chunk["text"]
                    # NOTE: Removed similarity scores and char positions
                }
                final_chunks_topk.append(chunk_topk)
            
            logger.debug(
                f"Processed {doc_id}: {len(final_chunks_all)} chunks, "
                f"{len(final_chunks_topk)} top-k selected"
            )
            
            return (final_chunks_all, final_chunks_topk, doc_id, all_chunks_path)
            
        except KeyError as e:
            logger.error(f"Missing required field in document {doc_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
            return None
    
    def select_top_k_chunks(
        self,
        chunks: List[Dict],
        k: int,
        method: str = "doc_similarity"
    ) -> List[Dict]:
        """
        Select top k chunks based on similarity metric.
        
        Args:
            chunks: List of chunks with similarity scores
            k: Number of chunks to select
            method: 'doc_similarity' or 'avg_similarity'
            
        Returns:
            List of top k chunks
        """
        if len(chunks) <= k:
            return chunks
        
        # Sort by the specified method
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get(method, 0.0),
            reverse=True
        )
        
        return sorted_chunks[:k]
    
    def process_batch(
        self,
        documents: List[Dict],
        skip_errors: bool = True
    ) -> Tuple[List[Dict], List[Dict], Dict[str, str], List[str]]:
        """
        Process a batch of documents.
        
        Args:
            documents: List of documents to process
            skip_errors: Whether to skip failed documents
            
        Returns:
            Tuple of (all_chunks, top_k_chunks, doc_to_chunks_path_mapping, failed_doc_ids)
        """
        all_chunks = []
        top_k_chunks = []
        doc_to_chunks_path = {}  # Maps doc_id -> all_chunks_path
        failed_docs = []
        
        for doc in documents:
            result = self.process_document(doc)
            
            if result is None:
                source_path = doc.get("_source_file", "unknown")
                doc_id = generate_document_id_from_path(source_path)
                failed_docs.append(doc_id)
                
                if not skip_errors:
                    raise ValueError(f"Failed to process document {doc_id}")
            else:
                doc_all_chunks, doc_top_k, doc_id, chunks_path = result
                all_chunks.extend(doc_all_chunks)
                top_k_chunks.extend(doc_top_k)
                doc_to_chunks_path[doc_id] = chunks_path
        
        return all_chunks, top_k_chunks, doc_to_chunks_path, failed_docs
    
    def process_dataset(
        self,
        documents: List[Dict],
        skip_errors: bool = True,
        use_parallel: bool = True
    ) -> Tuple[List[Dict], List[Dict], Dict[str, str], SemanticProcessingStats]:
        """
        Process entire dataset of documents.
        
        Args:
            documents: List of documents
            skip_errors: Whether to skip failed documents
            use_parallel: Whether to use parallel processing
            
        Returns:
            Tuple of (all_chunks, top_k_chunks, doc_to_chunks_path, stats)
        """
        logger.info(f"Processing {len(documents)} documents...")
        
        start_time = time.time()
        stats = SemanticProcessingStats()
        stats.total_documents = len(documents)
        stats.batch_size = self.batch_size
        stats.num_workers = self.num_workers
        
        all_chunks = []
        top_k_chunks = []
        doc_to_chunks_path = {}
        
        # Split into batches
        batches = [
            documents[i:i + self.batch_size]
            for i in range(0, len(documents), self.batch_size)
        ]
        
        if use_parallel and self.num_workers > 1:
            # Parallel processing
            logger.info(f"Using parallel processing with {self.num_workers} workers")
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self.process_batch, batch, skip_errors): idx
                    for idx, batch in enumerate(batches)
                }
                
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing batches"
                ):
                    try:
                        batch_all, batch_topk, batch_paths, batch_failed = future.result()
                        all_chunks.extend(batch_all)
                        top_k_chunks.extend(batch_topk)
                        doc_to_chunks_path.update(batch_paths)
                        stats.failed_doc_ids.extend(batch_failed)
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        if not skip_errors:
                            raise
        else:
            # Sequential processing
            logger.info("Using sequential processing")
            
            for batch in tqdm(batches, desc="Processing batches"):
                try:
                    batch_all, batch_topk, batch_paths, batch_failed = self.process_batch(
                        batch,
                        skip_errors
                    )
                    all_chunks.extend(batch_all)
                    top_k_chunks.extend(batch_topk)
                    doc_to_chunks_path.update(batch_paths)
                    stats.failed_doc_ids.extend(batch_failed)
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    if not skip_errors:
                        raise
        
        # Calculate statistics
        end_time = time.time()
        stats.total_time = end_time - start_time
        stats.successful_documents = len(documents) - len(stats.failed_doc_ids)
        stats.failed_documents = len(stats.failed_doc_ids)
        stats.total_chunks = len(all_chunks)
        stats.total_top_k_chunks = len(top_k_chunks)
        stats.avg_chunks_per_doc = (
            stats.total_chunks / stats.successful_documents
            if stats.successful_documents > 0 else 0.0
        )
        stats.avg_time_per_doc = (
            stats.total_time / len(documents)
            if len(documents) > 0 else 0.0
        )
        
        logger.info(f"Processing complete: {stats.successful_documents}/{stats.total_documents} succeeded")
        logger.info(f"Total chunks: {stats.total_chunks}, Top-K: {stats.total_top_k_chunks}")
        
        return all_chunks, top_k_chunks, doc_to_chunks_path, stats


def make_json_safe(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif hasattr(obj, "item"):   # numpy scalars
        return obj.item()
    else:
        return obj


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """Configure logging for the pipeline."""
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    logger.info(f"Logging configured: level={level}, file={log_file}")


if __name__ == "__main__":
    # Test the ID generation
    test_paths = [
        "raw/newapp/decisions/queensland/2005/document.json",
        "raw/newapp/legislation/commonwealth/1906/act.json",
        "decisions/nsw/2020/case_12345.json"
    ]
    
    print("Testing document ID generation:")
    for path in test_paths:
        doc_id = generate_document_id_from_path(path)
        chunks_path = get_all_chunks_adls_path(path)
        print(f"\nSource: {path}")
        print(f"Doc ID: {doc_id}")
        print(f"Chunks path: {chunks_path}")