"""
Azure AI Search Uploader Module
Handles uploading processed chunks with embeddings to Azure AI Search.
"""
import json
import logging
import time
from typing import List, Dict, Optional
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SearchIndexManager:
    """Manages Azure AI Search index creation and updates."""
    
    def __init__(self, endpoint: str, key: str):
        """
        Initialize search index manager.
        
        Args:
            endpoint: Azure Search endpoint URL
            key: Azure Search admin key
        """
        self.endpoint = endpoint
        self.credential = AzureKeyCredential(key)
        self.index_client = SearchIndexClient(endpoint, self.credential)
        logger.info(f"Search index manager initialized for {endpoint}")
    
    def create_legal_documents_index(
        self,
        index_name: str,
        vector_dimensions: int = 384,
        force_recreate: bool = False
    ) -> bool:
        """
        Create or update the legal documents search index.
        
        Args:
            index_name: Name of the index
            vector_dimensions: Dimension of embedding vectors
            force_recreate: Whether to delete and recreate if exists
            
        Returns:
            True if created/updated successfully
        """
        logger.info(f"Creating/updating index: {index_name}")
        
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.index_client.list_indexes()]
            
            if index_name in existing_indexes:
                if force_recreate:
                    logger.info(f"Deleting existing index: {index_name}")
                    self.index_client.delete_index(index_name)
                else:
                    logger.info(f"Index {index_name} already exists, skipping creation")
                    return True
            
            # Define fields
            fields = [
                # Key field
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True
                ),
                
                # Core metadata
                SimpleField(
                    name="chunk_id",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                SimpleField(
                    name="version_id",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                SimpleField(
                    name="type",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                SimpleField(
                    name="jurisdiction",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                SimpleField(
                    name="date",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    sortable=True
                ),
                
                # Structural metadata
                SimpleField(
                    name="start_char",
                    type=SearchFieldDataType.Int32,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="end_char",
                    type=SearchFieldDataType.Int32,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="num_sentences",
                    type=SearchFieldDataType.Int32,
                    filterable=True
                ),
                
                # Similarity metadata
                SimpleField(
                    name="avg_similarity",
                    type=SearchFieldDataType.Double,
                    sortable=True
                ),
                SimpleField(
                    name="doc_similarity",
                    type=SearchFieldDataType.Double,
                    sortable=True
                ),
                
                # Searchable text
                SearchableField(
                    name="citation",
                    type=SearchFieldDataType.String,
                    analyzer_name="standard.lucene"
                ),
                SearchableField(
                    name="text",
                    type=SearchFieldDataType.String,
                    analyzer_name="standard.lucene"
                ),
                SimpleField(
                    name="role",
                    type= SearchFieldDataType.String,
                    filterable = True
                ),
                SimpleField(
                    name="confidence",
                    type= SearchFieldDataType.Double,
                    filterable = False
                ),
                # Source file (optional)
                SimpleField(
                    name="source_file",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                
                # Vector field
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=vector_dimensions,
                    vector_search_profile_name="vector-profile"
                )
            ]
            
            # Define vector search configuration
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="hnsw-config"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-config",
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric="cosine"
                        )
                    )
                ]
            )
            
            # Create index
            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            result = self.index_client.create_or_update_index(index)
            logger.info(f"Index '{index_name}' created/updated successfully")
            return True
            
        except AzureError as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def delete_index(self, index_name: str) -> bool:
        """Delete an index."""
        try:
            self.index_client.delete_index(index_name)
            logger.info(f"Index '{index_name}' deleted")
            return True
        except AzureError as e:
            logger.error(f"Failed to delete index: {e}")
            return False
    
    def index_exists(self, index_name: str) -> bool:
        """Check if index exists."""
        try:
            existing = [idx.name for idx in self.index_client.list_indexes()]
            return index_name in existing
        except AzureError as e:
            logger.error(f"Error checking index existence: {e}")
            return False


class SearchUploader:
    """Uploads documents with embeddings to Azure AI Search."""
    
    def __init__(
        self,
        endpoint: str,
        key: str,
        index_name: str,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize search uploader.
        
        Args:
            endpoint: Azure Search endpoint URL
            key: Azure Search admin key
            index_name: Name of the index
            batch_size: Number of documents per batch
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.endpoint = endpoint
        self.index_name = index_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.credential = AzureKeyCredential(key)
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=self.credential
        )
        
        logger.info(
            f"Search uploader initialized: index={index_name}, "
            f"batch_size={batch_size}"
        )
    
    def prepare_document(self, chunk: Dict) -> Dict:
        """
        Prepare a chunk document for upload to Azure Search.
        
        Args:
            chunk: Chunk dictionary with metadata and embedding
            
        Returns:
            Prepared document for upload
        """
        # Create unique ID
        doc_id = chunk.get("id") or chunk.get("chunk_id", "unknown")
        
        # Build document
        doc = {
            "id": doc_id.replace("/", "_").replace(" ", "_"),  # Ensure valid ID
            "chunk_id": chunk.get("chunk_id", ""),
            "version_id": chunk.get("version_id", ""),
            "type": chunk.get("type", ""),
            "jurisdiction": chunk.get("jurisdiction", ""),
            "date": chunk.get("date", ""),
            "start_char": chunk.get("start_char", 0),
            "end_char": chunk.get("end_char", 0),
            "num_sentences": chunk.get("num_sentences", 0),
            "avg_similarity": chunk.get("avg_similarity", 0.0),
            "doc_similarity": chunk.get("doc_similarity", 0.0),
            "citation": chunk.get("citation", ""),
            "text": chunk.get("text", ""),
            "source_file": chunk.get("_source_file", ""),
            "embedding": chunk.get("embedding", [])
        }
        if "role_prediction" in chunk:
            doc['role'] = chunk["role_prediction"]["role"]
            doc['role_confidence'] = chunk["role_prediction"]["confidence"]
        return doc
    
    def upload_batch(
        self,
        documents: List[Dict],
        retry_count: int = 0
    ) -> bool:
        """
        Upload a batch of documents to Azure Search.
        
        Args:
            documents: List of prepared documents
            retry_count: Current retry attempt
            
        Returns:
            True if successful
        """
        try:
            result = self.search_client.upload_documents(documents=documents)
            
            # Check for failures
            failed = [r for r in result if not r.succeeded]
            
            if failed:
                logger.warning(
                    f"Batch upload partially failed: {len(failed)}/{len(documents)} documents"
                )
                for failure in failed[:5]:  # Log first 5 failures
                    logger.warning(f"  Failed: {failure.key} - {failure.error_message}")
                
                # Retry if not max attempts
                if retry_count < self.max_retries:
                    logger.info(f"Retrying failed documents (attempt {retry_count + 1})")
                    time.sleep(self.retry_delay)
                    failed_docs = [documents[i] for i, r in enumerate(result) if not r.succeeded]
                    return self.upload_batch(failed_docs, retry_count + 1)
                
                return False
            
            logger.debug(f"Batch of {len(documents)} documents uploaded successfully")
            return True
            
        except AzureError as e:
            logger.error(f"Batch upload failed: {e}")
            
            if retry_count < self.max_retries:
                logger.info(f"Retrying batch (attempt {retry_count + 1})")
                time.sleep(self.retry_delay)
                return self.upload_batch(documents, retry_count + 1)
            
            return False
    
    def upload_chunks(
        self,
        chunks: List[Dict],
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Upload all chunks to Azure Search in batches.
        
        Args:
            chunks: List of chunk dictionaries with embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"Uploading {len(chunks)} chunks to Azure Search")
        
        # Prepare documents
        documents = [self.prepare_document(chunk) for chunk in chunks]
        
        total_docs = len(documents)
        uploaded = 0
        failed = 0
        
        # Upload in batches
        iterator = range(0, total_docs, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Uploading batches")
        
        for i in iterator:
            batch = documents[i:i + self.batch_size]
            
            if self.upload_batch(batch):
                uploaded += len(batch)
            else:
                failed += len(batch)
        
        stats = {
            "total": total_docs,
            "uploaded": uploaded,
            "failed": failed,
            "success_rate": (uploaded / total_docs * 100) if total_docs > 0 else 0
        }
        
        logger.info(
            f"Upload complete: {uploaded}/{total_docs} documents uploaded "
            f"({stats['success_rate']:.1f}% success rate)"
        )
        
        return stats
    
    def delete_all_documents(self) -> bool:
        """Delete all documents from the index."""
        try:
            # This is a placeholder - Azure Search doesn't have a "delete all" API
            # You would need to query all document IDs and delete them
            logger.warning("Delete all documents not implemented - recreate index instead")
            return False
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False


def upload_to_search(
    chunks: List[Dict],
    endpoint: str,
    key: str,
    index_name: str,
    batch_size: int = 100,
    create_index: bool = False,
    vector_dimensions: int = 384
) -> Dict[str, int]:
    """
    Convenience function to upload chunks to Azure Search.
    
    Args:
        chunks: List of chunks with embeddings
        endpoint: Azure Search endpoint
        key: Azure Search admin key
        index_name: Index name
        batch_size: Upload batch size
        create_index: Whether to create index if not exists
        vector_dimensions: Embedding dimensions
        
    Returns:
        Upload statistics
    """
    # Create index if requested
    if create_index:
        manager = SearchIndexManager(endpoint, key)
        manager.create_legal_documents_index(index_name, vector_dimensions)
    
    # Upload documents
    uploader = SearchUploader(endpoint, key, index_name, batch_size)
    return uploader.upload_chunks(chunks)


if __name__ == "__main__":
    # Test with environment variables
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test index creation
    try:
        endpoint = os.getenv("SEARCH_ENDPOINT")
        key = os.getenv("SEARCH_KEY")
        
        if not endpoint or not key:
            raise ValueError("SEARCH_ENDPOINT and SEARCH_KEY environment variables are required")
        
        manager = SearchIndexManager(
            endpoint=endpoint,
            key=key
        )
        
        index_name = os.getenv("INDEX_NAME", "test-index")
        
        # Create index
        success = manager.create_legal_documents_index(
            index_name=index_name,
            vector_dimensions=384,
            force_recreate=False
        )
        
        if success:
            print(f"\n✓ Index '{index_name}' is ready")
        else:
            print(f"\n✗ Failed to create index '{index_name}'")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
