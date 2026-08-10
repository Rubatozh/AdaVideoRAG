import asyncio
import csv
import html
import io
import json
import logging
import os
import re
import tempfile
import threading
import weakref
from dataclasses import dataclass
from functools import lru_cache, wraps
from hashlib import md5
from typing import Any, Callable


import numpy as np
import tiktoken

logger = logging.getLogger("adavideorag")


@lru_cache(maxsize=None)
def _get_tiktoken_encoder(model_name: str):
    return tiktoken.encoding_for_model(model_name)

def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    tokens = _get_tiktoken_encoder(model_name).encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    content = _get_tiktoken_encoder(model_name).decode(tokens)
    return content


def truncate_list_by_token_size(
    list_data: list,
    key: Callable,
    max_token_size: int,
    model_name: str = "gpt-4o",
):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data), model_name=model_name))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def write_json(json_obj, file_name):
    target = os.path.abspath(file_name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.",
        suffix=".tmp",
        dir=os.path.dirname(target),
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(json_obj, output, indent=2, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, target)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


# it's dirty to type, so it's a good way to have fun
def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def list_of_list_to_csv(data: list[list]):
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerows(data)
    return output.getvalue().rstrip("\n")


# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(value: Any) -> Any:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(value, str):
        return value

    result = html.unescape(value.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


# Utils types -----------------------------------------------------------------------
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    model_name: str
    func: Callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        # Had to fix this as the embedding function took only one named argument put it's passed in
        # positionally, now we need to pass both
        kwargs['model_name'] = self.model_name

        # If there are positional arguments, convert them to keyword arguments
        if args:
            # Assuming the first positional argument is always 'texts'
            if len(args) == 1 and isinstance(args[0], list):
                kwargs['texts'] = args[0]
            else:
                raise ValueError("Unexpected positional arguments. Expected a single list of texts")
        # Call the function with the updated keyword arguments
        return await self.func(**kwargs)


# Decorators ------------------------------------------------------------------------
def limit_async_func_call(max_size: int):
    """Limit concurrent calls to an async function."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    def final_decro(func):
        semaphores = weakref.WeakKeyDictionary()
        semaphores_lock = threading.Lock()

        def semaphore_for_running_loop():
            loop = asyncio.get_running_loop()
            with semaphores_lock:
                semaphore_ref = semaphores.get(loop)
                semaphore = (
                    semaphore_ref() if semaphore_ref is not None else None
                )
                if semaphore is None:
                    semaphore = asyncio.Semaphore(max_size)
                    semaphores[loop] = weakref.ref(semaphore)
            return semaphore

        @wraps(func)
        async def wait_func(*args, **kwargs):
            semaphore = semaphore_for_running_loop()
            async with semaphore:
                return await func(*args, **kwargs)

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro
