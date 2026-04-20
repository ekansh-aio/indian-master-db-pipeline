"""
Azure AI Search Uploader Module
Handles uploading processed chunks with embeddings to Azure AI Search.
Supports SC and HC document schemas (doc_type 0=HC, 1=SC).
"""
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


def _base_fields(vector_dimensions: int) -> list:
    """Fields common to both SC and HC indices."""
    return [
        SimpleField(name="id",             type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name="doc_id",         type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="jurisdiction",   type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="date",           type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="start_char",     type=SearchFieldDataType.Int32,  filterable=True, sortable=True),
        SimpleField(name="end_char",       type=SearchFieldDataType.Int32,  filterable=True, sortable=True),
        SimpleField(name="num_sentences",  type=SearchFieldDataType.Int32,  filterable=True),
        SimpleField(name="avg_similarity", type=SearchFieldDataType.Double, sortable=True),
        SimpleField(name="doc_similarity", type=SearchFieldDataType.Double, sortable=True),
        SimpleField(name="role",           type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="confidence",     type=SearchFieldDataType.Double),
        SearchableField(name="title",      type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchableField(name="text",       type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name="vector-profile"
        )
    ]


def _sc_fields() -> list:
    """Extra fields for Supreme Court index."""
    return [
        SimpleField(name="source_file",    type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="all_chunks_path", type=SearchFieldDataType.String)
    ]


def _hc_fields() -> list:
    """Extra fields for High Court index."""
    return [
        SimpleField(name="judge",          type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="pdf_link",       type=SearchFieldDataType.String),
        SimpleField(name="cnr",            type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="court",          type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="all_chunks_path", type=SearchFieldDataType.String)
    ]


class SearchIndexManager:
    """Manages Azure AI Search index creation and updates."""

    def __init__(self, endpoint: str, key: str):
        self.credential = AzureKeyCredential(key)
        self.index_client = SearchIndexClient(endpoint, self.credential)
        logger.info(f"Search index manager initialized for {endpoint}")

    def create_legal_documents_index(
        self,
        index_name: str,
        doc_type: int,
        vector_dimensions: int = 384,
        force_recreate: bool = False
    ) -> bool:
        """
        Create or update the legal documents search index.

        Args:
            index_name:        Name of the index
            doc_type:          0 = High Court, 1 = Supreme Court
            vector_dimensions: Dimension of embedding vectors
            force_recreate:    Delete and recreate if index already exists
        """
        logger.info(f"Creating index '{index_name}' (doc_type={'SC' if doc_type == 1 else 'HC'})")

        try:
            existing = [idx.name for idx in self.index_client.list_indexes()]

            if index_name in existing:
                if force_recreate:
                    logger.info(f"Deleting existing index: {index_name}")
                    self.index_client.delete_index(index_name)
                else:
                    logger.info(f"Index '{index_name}' already exists, skipping creation")
                    return True

            fields = _base_fields(vector_dimensions)
            fields += _sc_fields() if doc_type == 1 else _hc_fields()

            vector_search = VectorSearch(
                profiles=[VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )],
                algorithms=[HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters=HnswParameters(m=4, ef_construction=400, ef_search=500, metric="cosine")
                )]
            )

            index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
            self.index_client.create_or_update_index(index)
            logger.info(f"Index '{index_name}' created successfully")
            return True

        except AzureError as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def delete_index(self, index_name: str) -> bool:
        try:
            self.index_client.delete_index(index_name)
            logger.info(f"Index '{index_name}' deleted")
            return True
        except AzureError as e:
            logger.error(f"Failed to delete index: {e}")
            return False

    def index_exists(self, index_name: str) -> bool:
        try:
            return index_name in [idx.name for idx in self.index_client.list_indexes()]
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
        doc_type: int,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Args:
            doc_type: 0 = High Court, 1 = Supreme Court — controls which fields
                      are included in each uploaded document.
        """
        self.index_name = index_name
        self.doc_type = doc_type
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        credential = AzureKeyCredential(key)
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential
        )

        logger.info(f"SearchUploader ready: index={index_name}, doc_type={'SC' if doc_type == 1 else 'HC'}")

    def prepare_document(self, chunk: Dict) -> Dict:
        """Map a chunk dict to an Azure Search document."""
        doc_id = chunk.get("id") or chunk.get("chunk_id", "unknown")

        doc = {
            "id":            doc_id.replace("/", "_").replace(" ", "_"),
            "doc_id":        chunk.get("doc_id", ""),
            "jurisdiction":  chunk.get("jurisdiction", ""),
            "date":          chunk.get("date", ""),
            "start_char":    chunk.get("start_char", 0),
            "end_char":      chunk.get("end_char", 0),
            "num_sentences": chunk.get("num_sentences", 0),
            "avg_similarity": chunk.get("avg_similarity", 0.0),
            "doc_similarity": chunk.get("doc_similarity", 0.0),
            "role":          chunk.get("role", ""),
            "confidence":    chunk.get("confidence", 0.0),
            "title":         chunk.get("title", ""),
            "text":          chunk.get("text", ""),
            "embedding":     chunk.get("embedding", [])
        }

        if self.doc_type == 1:  # Supreme Court
            doc["source_file"]     = chunk.get("original_source_path", "")
            doc["all_chunks_path"] = chunk.get("all_chunks_path", "")
        else:                   # High Court
            doc["judge"]           = chunk.get("judge", "")
            doc["pdf_link"]        = chunk.get("pdf_link", "")
            doc["cnr"]             = chunk.get("cnr", "")
            doc["court"]           = chunk.get("court", "")
            doc["all_chunks_path"] = chunk.get("all_chunks_path", "")

        return doc

    def upload_batch(self, documents: List[Dict], retry_count: int = 0) -> bool:
        try:
            result = self.search_client.upload_documents(documents=documents)
            failed = [r for r in result if not r.succeeded]

            if failed:
                logger.warning(f"Batch partially failed: {len(failed)}/{len(documents)}")
                for r in failed[:5]:
                    logger.warning(f"  Failed: {r.key} — {r.error_message}")

                if retry_count < self.max_retries:
                    time.sleep(self.retry_delay)
                    failed_docs = [documents[i] for i, r in enumerate(result) if not r.succeeded]
                    return self.upload_batch(failed_docs, retry_count + 1)
                return False

            logger.debug(f"Uploaded batch of {len(documents)}")
            return True

        except AzureError as e:
            logger.error(f"Batch upload failed: {e}")
            if retry_count < self.max_retries:
                time.sleep(self.retry_delay)
                return self.upload_batch(documents, retry_count + 1)
            return False

    def upload_chunks(self, chunks: List[Dict], show_progress: bool = True) -> Dict[str, int]:
        logger.info(f"Uploading {len(chunks)} chunks to '{self.index_name}'")

        documents = [self.prepare_document(chunk) for chunk in chunks]
        total = len(documents)
        uploaded = 0
        failed = 0

        iterator = range(0, total, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Uploading batches")

        for i in iterator:
            batch = documents[i:i + self.batch_size]
            if self.upload_batch(batch):
                uploaded += len(batch)
            else:
                failed += len(batch)

        stats = {
            "total": total,
            "uploaded": uploaded,
            "failed": failed,
            "success_rate": (uploaded / total * 100) if total > 0 else 0
        }

        logger.info(
            f"Upload complete: {uploaded}/{total} "
            f"({stats['success_rate']:.1f}% success)"
        )
        return stats
