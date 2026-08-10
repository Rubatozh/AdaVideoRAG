import asyncio
import os
import threading
from dataclasses import dataclass, field

import numpy as np
import torch
from nano_vectordb import NanoVectorDB
from tqdm import tqdm
from imagebind.models import imagebind_model

from .._utils import logger
from ..base import BaseVectorStorage
from .._videoutil import encode_video_segments, encode_string_query


async def _embed_contents(contents, embedding_func, batch_size):
    if batch_size <= 0:
        raise ValueError("embedding batch size must be positive")
    batches = [
        contents[index : index + batch_size]
        for index in range(0, len(contents), batch_size)
    ]
    embeddings = np.concatenate(
        await asyncio.gather(*[embedding_func(batch) for batch in batches])
    )
    if len(embeddings) != len(contents):
        raise ValueError(
            "Embedding endpoint returned an unexpected number of vectors: "
            f"expected {len(contents)}, got {len(embeddings)}"
        )
    return embeddings


@dataclass
class NanoVectorDBAndVideoStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["llm"]["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__video_id__": v['video_segment_id'],
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        embeddings = await _embed_contents(
            contents,
            self.embedding_func,
            self._max_batch_size,
        )
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "video_id": dp["__video_id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        await asyncio.to_thread(self._client.save)

@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["llm"]["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        embeddings = await _embed_contents(
            contents,
            self.embedding_func,
            self._max_batch_size,
        )
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        await asyncio.to_thread(self._client.save)


@dataclass
class NanoVectorDBVideoSegmentStorage(BaseVectorStorage):
    embedding_func = None
    segment_retrieval_top_k: int = 2
    cosine_better_than_threshold: float = 0.5
    _embedder: object = field(default=None, init=False, repr=False)
    _embedder_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["video_embedding_batch_num"]
        self._client = NanoVectorDB(
            self.global_config["video_embedding_dim"], storage_file=self._client_file_name
        )
        self.top_k = self.global_config.get(
            "segment_retrieval_top_k", self.segment_retrieval_top_k
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "visual_query_better_than_threshold",
            self.cosine_better_than_threshold,
        )

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        with self._embedder_lock:
            if self._embedder is None:
                configured_device = self.global_config.get("cuda")
                if configured_device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                elif str(configured_device).startswith(("cuda", "cpu")):
                    device = str(configured_device)
                else:
                    device = f"cuda:{configured_device}"
                if device.startswith("cuda") and not torch.cuda.is_available():
                    logger.warning("CUDA is unavailable; loading ImageBind on CPU")
                    device = "cpu"
                self._embedder = imagebind_model.imagebind_huge(pretrained=True).to(
                    device
                )
                self._embedder.eval()
        return self._embedder

    async def upsert(self, video_name, segment_index2name, video_output_format):
        embedder = await asyncio.to_thread(self._get_embedder)

        logger.info(f"Inserting {len(segment_index2name)} segments to {self.namespace}")
        if not len(segment_index2name):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data, video_paths = [], []
        cache_path = os.path.join(self.global_config["working_dir"], '_cache', video_name)
        index_list = list(segment_index2name.keys())
        for index in index_list:
            list_data.append({
                "__id__": f"{video_name}_{index}",
                "__video_name__": video_name,
                "__index__": index,
            })
            segment_name = segment_index2name[index]
            video_file = os.path.join(cache_path, f"{segment_name}.{video_output_format}")
            video_paths.append(video_file)
        batches = [
            video_paths[i: i + self._max_batch_size]
            for i in range(0, len(video_paths), self._max_batch_size)
        ]
        embeddings = []
        for _batch in tqdm(batches, desc=f"Encoding Video Segments {video_name}"):
            batch_embeddings = await asyncio.to_thread(
                encode_video_segments, _batch, embedder
            )
            embeddings.append(batch_embeddings)
        embeddings = torch.concat(embeddings, dim=0)
        embeddings = embeddings.numpy()
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str):
        embedder = await asyncio.to_thread(self._get_embedder)

        embedding = await asyncio.to_thread(encode_string_query, query, embedder)
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=self.top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        await asyncio.to_thread(self._client.save)
