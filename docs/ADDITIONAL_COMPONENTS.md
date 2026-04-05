# Additional Pipeline Components Guide

## Components You Already Have ✅

Your current pipeline implementation includes:

1. **Data Ingestion Layer** (`adls_fetcher.py`)
   - Fetches data from Azure Data Lake Storage
   - Supports batch and streaming modes
   - Handles nested folder structures

2. **Text Preprocessing** (`legal_text_cleaner.py`)
   - Removes headers, citations, formatting noise
   - Normalizes whitespace
   - Optimized with compiled regex patterns

3. **Semantic Chunking** (`semantic_chunker.py`)
   - Sentence-level semantic splitting
   - Similarity-based grouping
   - Chunk-to-document relevance scoring
   - Top-K selection

4. **Embedding Generation** (via sentence-transformers)
   - Batch processing support
   - 384-dimensional vectors
   - Reuses model across chunks

5. **Vector Index** (`search_uploader.py`)
   - Azure AI Search with HNSW algorithm
   - Metadata filtering support
   - Batch upload with retry logic

6. **Orchestration Layer** (`pipeline.py`)
   - End-to-end workflow management
   - Error handling and logging
   - Statistics tracking

## Missing Components for Production RAG Systems

Here are standard components you should consider building:

---

## 1. Query Engine (Critical - Next Priority)

**Purpose:** Retrieve relevant chunks for user queries

**File:** `query_engine.py`

**Key Features:**
- Vector similarity search
- Hybrid search (vector + keyword)
- Metadata filtering
- Result re-ranking
- Query expansion

**Sample Implementation:**

```python
class QueryEngine:
    def __init__(self, search_client, embedding_model):
        self.search_client = search_client
        self.embedding_model = embedding_model
    
    def search(self, query: str, top_k: int = 10, filters: Dict = None):
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Hybrid search (vector + keyword)
        results = self.search_client.search(
            search_text=query,  # Keyword search
            vector_queries=[VectorQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k
            )],
            filter=filters  # e.g., jurisdiction='queensland'
        )
        
        return results
    
    def rerank(self, query: str, results: List, cross_encoder):
        # Use cross-encoder for better ranking
        pairs = [(query, r['text']) for r in results]
        scores = cross_encoder.predict(pairs)
        
        # Re-sort by cross-encoder scores
        reranked = sorted(zip(results, scores), 
                         key=lambda x: x[1], reverse=True)
        return [r for r, s in reranked]
```

**Priority:** HIGH - This is needed to actually retrieve documents

---

## 2. Evaluation Framework (Important)

**Purpose:** Measure retrieval quality

**File:** `evaluation.py`

**Key Metrics:**
- **MRR** (Mean Reciprocal Rank)
- **NDCG** (Normalized Discounted Cumulative Gain)
- **Recall@K** (What % of relevant docs are in top K)
- **Precision@K**
- **Hit Rate**

**Sample Implementation:**

```python
def evaluate_retrieval(
    test_queries: List[Dict],  # {query, relevant_doc_ids}
    query_engine: QueryEngine
) -> Dict[str, float]:
    
    mrr_scores = []
    recall_at_10 = []
    
    for item in test_queries:
        query = item['query']
        relevant_ids = set(item['relevant_doc_ids'])
        
        # Retrieve
        results = query_engine.search(query, top_k=10)
        retrieved_ids = [r['id'] for r in results]
        
        # Calculate MRR
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                mrr_scores.append(1.0 / rank)
                break
        
        # Calculate Recall@10
        hits = len(relevant_ids & set(retrieved_ids))
        recall = hits / len(relevant_ids) if relevant_ids else 0
        recall_at_10.append(recall)
    
    return {
        'MRR': np.mean(mrr_scores),
        'Recall@10': np.mean(recall_at_10)
    }
```

**Priority:** HIGH - Needed to know if your system works well

---

## 3. Incremental Processing (Production Essential)

**Purpose:** Process only new/updated documents

**File:** `incremental_processor.py`

**Key Features:**
- Track processed documents in database/file
- Compare file timestamps
- Delta processing
- Version control

**Sample Implementation:**

```python
class IncrementalProcessor:
    def __init__(self, state_file='processed_docs.json'):
        self.state_file = state_file
        self.processed = self.load_state()
    
    def load_state(self) -> Dict:
        if Path(self.state_file).exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}
    
    def is_processed(self, doc_id: str, doc_hash: str) -> bool:
        return self.processed.get(doc_id) == doc_hash
    
    def mark_processed(self, doc_id: str, doc_hash: str):
        self.processed[doc_id] = doc_hash
        with open(self.state_file, 'w') as f:
            json.dump(self.processed, f)
    
    def filter_new_docs(self, documents: List[Dict]) -> List[Dict]:
        new_docs = []
        for doc in documents:
            doc_id = doc['id']
            doc_hash = hashlib.md5(doc['text'].encode()).hexdigest()
            
            if not self.is_processed(doc_id, doc_hash):
                new_docs.append(doc)
        
        return new_docs
```

**Priority:** MEDIUM - Critical for production, but not for initial testing

---

## 4. Monitoring & Observability (Production Essential)

**Purpose:** Track pipeline health and performance

**File:** `monitoring.py`

**Key Metrics:**
- Processing throughput (docs/second)
- Error rates
- Latency percentiles (p50, p95, p99)
- Queue depths
- Cost tracking (Azure API calls)

**Sample Implementation:**

```python
import time
from dataclasses import dataclass
from typing import Dict
import psutil

@dataclass
class PipelineMetrics:
    docs_processed: int = 0
    docs_failed: int = 0
    chunks_generated: int = 0
    embeddings_generated: int = 0
    start_time: float = None
    
    def record_document(self, success: bool, num_chunks: int):
        if success:
            self.docs_processed += 1
            self.chunks_generated += num_chunks
        else:
            self.docs_failed += 1
    
    def get_throughput(self) -> float:
        if not self.start_time:
            return 0
        elapsed = time.time() - self.start_time
        return self.docs_processed / elapsed if elapsed > 0 else 0
    
    def get_summary(self) -> Dict:
        return {
            'documents_processed': self.docs_processed,
            'documents_failed': self.docs_failed,
            'chunks_generated': self.chunks_generated,
            'throughput_docs_per_sec': self.get_throughput(),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }
```

**Priority:** MEDIUM - Important for production operations

---

## 5. Data Validation Layer (Quality Assurance)

**Purpose:** Ensure data quality throughout pipeline

**File:** `validators.py`

**Key Checks:**
- Schema validation
- Required fields present
- Text length validation
- Embedding dimension validation
- Duplicate detection

**Sample Implementation:**

```python
from typing import List, Dict

class DataValidator:
    def __init__(self, required_fields: List[str]):
        self.required_fields = required_fields
    
    def validate_document(self, doc: Dict) -> tuple[bool, str]:
        # Check required fields
        for field in self.required_fields:
            if field not in doc:
                return False, f"Missing required field: {field}"
        
        # Check text is not empty
        if not doc.get('text', '').strip():
            return False, "Empty text field"
        
        # Check text length
        if len(doc['text']) < 50:
            return False, "Text too short"
        
        return True, "Valid"
    
    def validate_chunk(self, chunk: Dict) -> tuple[bool, str]:
        # Check embedding exists and has correct dimensions
        if 'embedding' not in chunk:
            return False, "Missing embedding"
        
        if len(chunk['embedding']) != 384:
            return False, f"Invalid embedding dimension: {len(chunk['embedding'])}"
        
        # Check chunk has text
        if not chunk.get('text', '').strip():
            return False, "Empty chunk text"
        
        return True, "Valid"
    
    def detect_duplicates(self, chunks: List[Dict]) -> List[tuple]:
        seen = {}
        duplicates = []
        
        for chunk in chunks:
            text_hash = hash(chunk['text'])
            if text_hash in seen:
                duplicates.append((chunk['id'], seen[text_hash]))
            else:
                seen[text_hash] = chunk['id']
        
        return duplicates
```

**Priority:** MEDIUM - Important for data quality

---

## 6. Caching Layer (Performance Optimization)

**Purpose:** Avoid recomputing expensive operations

**File:** `cache_manager.py`

**What to Cache:**
- Document embeddings (by hash)
- Cleaned text (by document ID)
- Query results (by query + filters)
- Model outputs

**Sample Implementation:**

```python
import hashlib
from pathlib import Path
import pickle

class CacheManager:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get_embedding(self, text: str):
        key = self.get_cache_key(text)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_embedding(self, text: str, embedding):
        key = self.get_cache_key(text)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)
    
    def clear_cache(self):
        for file in self.cache_dir.glob('*.pkl'):
            file.unlink()
```

**Priority:** LOW - Nice to have, but not essential initially

---

## 7. API Layer (If External Access Needed)

**Purpose:** Expose pipeline as REST API

**File:** `api_server.py`

**Framework:** FastAPI

**Endpoints:**
- `POST /query` - Search for documents
- `POST /process` - Process new document
- `GET /stats` - Get pipeline statistics
- `GET /health` - Health check

**Sample Implementation:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: dict = None

class ProcessRequest(BaseModel):
    document: dict

@app.post("/query")
async def search(request: QueryRequest):
    try:
        results = query_engine.search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_document(request: ProcessRequest):
    try:
        # Process document through pipeline
        result = pipeline.process_single_document(request.document)
        return {"status": "success", "chunks": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Priority:** LOW - Only if you need external API access

---

## 8. Export & Migration Tools

**Purpose:** Backup and migrate data

**File:** `export_manager.py`

**Features:**
- Export index to JSON
- Import from JSON
- Backup to blob storage
- Migration between indexes

**Priority:** LOW - Useful for maintenance

---

## 9. Reranking Layer (Quality Enhancement)

**Purpose:** Improve search relevance

**File:** `reranker.py`

**Models:**
- Cross-encoders (e.g., `ms-marco-MiniLM-L-12-v2`)
- More accurate but slower than bi-encoders

**Sample Implementation:**

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-12-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10):
        # Create query-document pairs
        pairs = [(query, doc['text']) for doc in documents]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Sort by score
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_scores[:top_k]]
```

**Priority:** MEDIUM - Significant quality improvement

---

## Recommended Implementation Order

### Phase 1 (Immediate - Core Functionality)
1. ✅ Data ingestion (done)
2. ✅ Processing pipeline (done)
3. ✅ Indexing (done)
4. **Query Engine** - BUILD THIS NEXT

### Phase 2 (Testing & Quality)
5. **Evaluation Framework** - Measure performance
6. **Data Validation** - Ensure quality
7. **Monitoring** - Track health

### Phase 3 (Production Readiness)
8. **Incremental Processing** - Handle updates efficiently
9. **Caching** - Improve performance
10. **Reranking** - Enhance quality

### Phase 4 (Optional/As Needed)
11. API Layer - If external access needed
12. Export/Migration Tools - For maintenance

---

## Quick Start: Building Query Engine

Here's a minimal query engine to get you started:

```python
# query_engine.py
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from sentence_transformers import SentenceTransformer

class SimpleQueryEngine:
    def __init__(self, endpoint, key, index_name, model_name):
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(key)
        )
        self.model = SentenceTransformer(model_name)
    
    def search(self, query: str, top_k: int = 10):
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Vector search
        results = self.search_client.search(
            search_text=None,
            vector_queries=[{
                "kind": "vector",
                "vector": query_embedding,
                "k": top_k,
                "fields": "embedding"
            }]
        )
        
        # Return results
        return [
            {
                'text': r['text'],
                'score': r['@search.score'],
                'citation': r.get('citation', ''),
                'jurisdiction': r.get('jurisdiction', '')
            }
            for r in results
        ]

# Usage
if __name__ == "__main__":
    engine = SimpleQueryEngine(
        endpoint=os.getenv("SEARCH_ENDPOINT"),
        key=os.getenv("SEARCH_KEY"),
        index_name=os.getenv("INDEX_NAME"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    results = engine.search("contract interpretation", top_k=5)
    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(f"Text: {r['text'][:200]}...")
        print()
```

Save this as `query_engine.py` and test it after your pipeline runs!

---

## Summary

Your pipeline is well-structured for **ingestion and indexing**. To make it a complete RAG system, focus on:

1. **Query Engine** (critical) - Retrieve documents
2. **Evaluation** (important) - Measure quality
3. **Incremental Processing** (production) - Handle updates
4. **Monitoring** (production) - Track health

Everything else can be added as needed based on your requirements!
