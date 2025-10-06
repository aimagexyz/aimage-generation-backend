from typing import BinaryIO

from aiocache import cached
from asgiref.sync import sync_to_async
from fastapi.encoders import jsonable_encoder
from vertexai.language_models import TextEmbeddingInput
from vertexai.vision_models import Image

from aimage_supervision.clients.google import multimodal_embedding_model, text_embedding_model


def _get_text_embedding(text: str, dimensionality: int = 768) -> list[float]:
    """获取文本的嵌入向量"""
    inputs = [TextEmbeddingInput(text, task_type="RETRIEVAL_DOCUMENT")]
    kwargs = dict(
        output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = text_embedding_model.get_embeddings(inputs, **kwargs)
    return embeddings[0].values


def _get_multimodal_embedding_text(text: str, dimension: int = 512) -> list[float]:
    """获取多模态文本的嵌入向量"""
    embedding = multimodal_embedding_model.get_embeddings(
        contextual_text=text,
        dimension=dimension,
    )
    return embedding.text_embedding


def _get_multimodal_embedding_image(image_bytes: bytes, dimension: int = 512) -> list[float]:
    """获取图片的嵌入向量"""
    image = Image(image_bytes=image_bytes)
    embedding = multimodal_embedding_model.get_embeddings(
        image=image,
        dimension=dimension,
    )
    return embedding.image_embedding


@cached(ttl=3600)
async def get_text_embedding(content: str) -> list[float]:
    """获取文本的嵌入向量（异步包装器）"""
    embeddings = await sync_to_async(_get_text_embedding)(content)
    return jsonable_encoder(embeddings)


@cached(ttl=3600)
async def get_multimodal_embedding_text(content: str) -> list[float]:
    """获取多模态文本的嵌入向量（异步包装器）"""
    embeddings = await sync_to_async(_get_multimodal_embedding_text)(content)
    return jsonable_encoder(embeddings)


@cached(ttl=3600)
async def get_multimodal_embedding_image(image_file: BinaryIO) -> list[float]:
    """获取图片的嵌入向量（异步包装器）"""
    image_bytes = await sync_to_async(image_file.read)()
    embeddings = await sync_to_async(_get_multimodal_embedding_image)(image_bytes)
    return jsonable_encoder(embeddings)
