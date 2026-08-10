import asyncio
import base64
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Callable, Optional

import numpy as np
from openai import AsyncOpenAI, APIConnectionError, RateLimitError
from PIL import Image
from qwen_vl_utils import process_vision_info
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._utils import EmbeddingFunc, compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage


global_openai_async_client = None

api_key = os.getenv("ADAVIDEORAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
base_url = os.getenv("ADAVIDEORAG_LLM_BASE_URL", "http://localhost:9997/v1")
IMAGE_EXTENSIONS = {".gif", ".jpeg", ".jpg", ".png", ".webp"}


def configure_openai(api_base=None, api_key_value=None):
    """Configure the OpenAI-compatible text and embedding endpoint."""
    global api_key, base_url, global_openai_async_client
    if api_base:
        base_url = api_base
    if api_key_value:
        api_key = api_key_value
    global_openai_async_client = None


def _response_text(response, operation: str) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError(f"{operation} returned no choices")
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{operation} returned empty text")
    return content


def prepare_message_for_vllm(content_messages):
    """Convert local visual media into vLLM-compatible base64 payloads."""
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content = message["content"]
        if not isinstance(message_content, list):
            vllm_messages.append(message)
            continue

        converted_content = []
        for part in message_content:
            if "video" in part:
                video_message = [{"content": [part]}]
                _, video_inputs, video_kwargs = process_vision_info(
                    video_message, return_video_kwargs=True
                )
                if video_inputs is None:
                    raise ValueError("Qwen video preprocessing returned no frames")
                video_input = (
                    video_inputs.pop()
                    .permute(0, 2, 3, 1)
                    .numpy()
                    .astype(np.uint8)
                )
                fps_list.extend(video_kwargs.get("fps", []))

                encoded_frames = []
                for frame in video_input:
                    output_buffer = BytesIO()
                    Image.fromarray(frame).save(output_buffer, format="jpeg")
                    encoded_frames.append(
                        base64.b64encode(output_buffer.getvalue()).decode("utf-8")
                    )
                part = {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/jpeg;base64,{','.join(encoded_frames)}"
                    },
                }
            elif "image" in part:
                image_source = part["image"]
                if isinstance(image_source, str) and image_source.startswith(
                    ("data:", "http://", "https://")
                ):
                    image_url = image_source
                else:
                    with Image.open(image_source) as source_image:
                        output_buffer = BytesIO()
                        source_image.convert("RGB").save(
                            output_buffer,
                            format="jpeg",
                        )
                    image_url = (
                        "data:image/jpeg;base64,"
                        + base64.b64encode(output_buffer.getvalue()).decode("utf-8")
                    )
                part = {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            converted_content.append(part)
        message["content"] = converted_content
        vllm_messages.append(message)
    return vllm_messages, {"fps": fps_list}


@dataclass
class QwenCalculatorvLLM_API:
    qwen_fps: float = 10
    api_key: str = field(
        default_factory=lambda: os.getenv("ADAVIDEORAG_VLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "ADAVIDEORAG_VLM_BASE_URL", "http://localhost:8000/v1"
        )
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("ADAVIDEORAG_VLM_MODEL", "Qwen2.5_VL")
    )

    def __post_init__(self):
        if self.qwen_fps <= 0:
            raise ValueError("qwen_fps must be positive")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def process_video(self, video_path, prompt):
        media_paths = (
            list(video_path)
            if isinstance(video_path, (list, tuple))
            else [video_path]
        )
        if not media_paths:
            raise ValueError("At least one image or video path is required")
        system_message = (
            "You are a helpful assistant. Answer using the supplied visual and text "
            "evidence. For a multiple-choice question, return only A, B, C, or D. "
            "For a summary question, return a concise answer. If multiple videos are "
            "provided, combine their evidence; for count and ordering questions, "
            "prefer the video evidence when it conflicts with text."
        )
        user_content = [{"type": "text", "text": prompt}]
        for index, path in enumerate(media_paths):
            extension = os.path.splitext(str(path))[1].lower()
            if extension in IMAGE_EXTENSIONS:
                user_content.append({"type": "image", "image": path})
            else:
                user_content.append(
                    {
                        "type": "video",
                        "video": path,
                        "total_pixels": 20480 * 28 * 28,
                        "min_pixels": 16 * 28 * 28,
                        # Preserve the released query sampling behavior.
                        "fps": (
                            5
                            if len(media_paths) > 1 and index == 0
                            else self.qwen_fps
                        ),
                    }
                )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]
        messages, video_kwargs = await asyncio.to_thread(
            prepare_message_for_vllm, messages
        )
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body={"mm_processor_kwargs": video_kwargs},
        )
        return _response_text(response, "Vision-language generation")


async def query_get_level(
    model_name,
    query,
    prompt,
    api_base=None,
    api_key_value=None,
):
    """Classify a query with a non-blocking OpenAI-compatible client."""
    async with AsyncOpenAI(
        base_url=api_base or base_url,
        api_key=api_key_value or api_key,
    ) as client:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )
    return _response_text(completion, "Intent classification")


def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return global_openai_async_client


@dataclass
class LLMConfig:
    embedding_func_raw: Callable
    embedding_model_name: str
    embedding_dim: int
    embedding_max_token_size: int
    embedding_batch_num: int
    embedding_func_max_async: int

    best_model_func_raw: Callable
    best_model_name: str
    best_model_max_token_size: int
    best_model_max_async: int

    cheap_model_func_raw: Callable
    cheap_model_name: str
    cheap_model_max_token_size: int
    cheap_model_max_async: int

    embedding_func: Optional[EmbeddingFunc] = None
    best_model_func: Optional[Callable] = None
    cheap_model_func: Optional[Callable] = None

    def __post_init__(self):
        positive_fields = {
            "embedding_dim": self.embedding_dim,
            "embedding_max_token_size": self.embedding_max_token_size,
            "embedding_batch_num": self.embedding_batch_num,
            "embedding_func_max_async": self.embedding_func_max_async,
            "best_model_max_token_size": self.best_model_max_token_size,
            "best_model_max_async": self.best_model_max_async,
            "cheap_model_max_token_size": self.cheap_model_max_token_size,
            "cheap_model_max_async": self.cheap_model_max_async,
        }
        invalid = [name for name, value in positive_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"LLMConfig fields must be positive: {', '.join(invalid)}")
        for name, function in (
            ("embedding_func_raw", self.embedding_func_raw),
            ("best_model_func_raw", self.best_model_func_raw),
            ("cheap_model_func_raw", self.cheap_model_func_raw),
        ):
            if not callable(function):
                raise TypeError(f"{name} must be callable")
        embedding_wrapper = wrap_embedding_func_with_attrs(
            embedding_dim=self.embedding_dim,
            max_token_size=self.embedding_max_token_size,
            model_name=self.embedding_model_name,
        )
        self.embedding_func = embedding_wrapper(self.embedding_func_raw)
        self.best_model_func = lambda prompt, *args, **kwargs: self.best_model_func_raw(
            self.best_model_name, prompt, *args, **kwargs
        )
        self.cheap_model_func = lambda prompt, *args, **kwargs: self.cheap_model_func_raw(
            self.cheap_model_name, prompt, *args, **kwargs
        )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages, kwargs)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    response = await client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    content = _response_text(response, "Text generation")
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": content, "model": model}})
    return content


async def gpt_4o_mini_complete(
    model_name, prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=float)
    client = get_openai_async_client_instance()
    response = await client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    if len(response.data) != len(texts):
        raise RuntimeError(
            "Embedding endpoint returned an unexpected number of vectors: "
            f"expected {len(texts)}, got {len(response.data)}"
        )
    return np.array([item.embedding for item in response.data])


openai_config = LLMConfig(
    embedding_func_raw=openai_embedding,
    embedding_model_name="text-embedding-3-small",
    embedding_dim=1536,
    embedding_max_token_size=8192,
    embedding_batch_num=32,
    embedding_func_max_async=16,
    best_model_func_raw=gpt_4o_mini_complete,
    best_model_name="deepseek_32B",
    best_model_max_token_size=32768,
    best_model_max_async=16,
    cheap_model_func_raw=gpt_4o_mini_complete,
    cheap_model_name="deepseek_32B",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=16,
)
