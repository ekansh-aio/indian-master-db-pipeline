import os
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from azure.search.documents.indexes.models import SearchField, SearchFieldDataType

load_dotenv()

endpoint = os.getenv("SEARCH_ENDPOINT")
key = os.getenv("SEARCH_KEY")
index_name = os.getenv("INDEX_NAME")

if endpoint is None:
    raise ValueError("SEARCH_ENDPOINT environment variable is required")
if key is None:
    raise ValueError("SEARCH_KEY environment variable is required")
if index_name is None:
    raise ValueError("INDEX_NAME environment variable is required")

client = SearchIndexClient(endpoint, AzureKeyCredential(key))

fields = [
    # Key
    SimpleField(name="id", type="Edm.String", key=True),

    # Core metadata
    SimpleField(name="chunk_id", type="Edm.String", filterable=True),
    SimpleField(name="version_id", type="Edm.String", filterable=True),
    SimpleField(name="type", type="Edm.String", filterable=True),
    SimpleField(name="jurisdiction", type="Edm.String", filterable=True),
    SimpleField(name="date", type="Edm.String", filterable=True, sortable=True),

    # Searchable text
    SearchableField(name="citation", type="Edm.String"),
    SearchableField(name="text", type="Edm.String"),

    # Vector field

    SearchField(
    name="embedding",
    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
    searchable=True,
    vector_search_dimensions=384,
    vector_search_profile_name="vector-profile"
    )

]

index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw")
        ]
    )
)

client.create_or_update_index(index)

print("✅ Vector + metadata index created successfully.")
