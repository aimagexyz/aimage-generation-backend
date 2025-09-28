import asyncio
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import NamedVector

from aimage_supervision.settings import QDRANT_API_KEY, QDRANT_URL
# Assuming this exists and is synchronous
from aimage_supervision.utils.embedding_utils import get_text_embeddings
from aimage_supervision.clients.aws_s3 import get_s3_url_from_path


async def get_similar_images(query_text: str, ip_name: str, data_type: str) -> List[str]:
    """
    Finds similar images based on a query text using Qdrant.
    Synchronous parts of the Qdrant client are run in a thread pool.
    """
    if not query_text:
        return []

    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Warning: QDRANT_URL or QDRANT_API_KEY not configured. Cannot get similar images.")
        return []

    db_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Actual type is List[ScoredPoint] from qdrant_client

    # Run text embedding generation (assuming it's synchronous and potentially blocking)
    embedding_values = get_text_embeddings([query_text])[0].values

    collection_name = f'{ip_name}-{data_type}'

    # Perform Qdrant search (synchronous)
    search_results = db_client.search(
        collection_name=collection_name,
        query_vector=NamedVector(
            name="description",  # Ensure this vector name matches your Qdrant collection setup
            vector=embedding_values,
        ),
        limit=10,
        with_vectors=False,
        with_payload=True,
    )

    similar_images: List[str] = []
    for result in search_results:
        # Ensure payload is a dictionary before accessing it
        if result.payload and isinstance(result.payload, dict) and "url" in result.payload:
            try:
                # get_s3_url_from_path is async and can be awaited directly
                s3_url = await get_s3_url_from_path(result.payload["url"], bucket_name='ai-mage-agi-review')
                similar_images.append(s3_url)
            except Exception as e:
                print(
                    f"Error getting S3 URL for path {result.payload['url']}: {e}")
        else:
            # Log or handle cases where payload is not as expected
            print(
                f"Warning: Qdrant result missing payload, URL in payload, or payload is not a dict: {result}")

    return similar_images
