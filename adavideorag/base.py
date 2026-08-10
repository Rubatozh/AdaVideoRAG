from dataclasses import dataclass, field
from typing import Generic, Literal, Mapping, TypeVar, TypedDict, Union

from ._utils import EmbeddingFunc


@dataclass
class QueryParam:
    mode: Literal["adavideorag"] = "adavideorag"
    top_k: int = 20
    naive_max_token_for_text_unit: int = 12000

    def __post_init__(self):
        if self.mode != "adavideorag":
            raise ValueError(f"Unknown query mode: {self.mode}")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.naive_max_token_for_text_unit <= 0:
            raise ValueError("naive_max_token_for_text_unit must be positive")


TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {
        "tokens": int,
        "content": str,
        "video_segment_id": Union[str, list[str]],
        "chunk_order_index": int,
    },
)

T = TypeVar("T")


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        return None


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    @property
    def data(self) -> Mapping[str, T]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def delete(self, ids: list[str]):
        """Remove records that belong to an uncommitted transaction."""
        raise NotImplementedError

@dataclass
class BaseGraphStorage(StorageNameSpace):
    def snapshot(self):
        raise NotImplementedError

    def restore(self, snapshot):
        raise NotImplementedError

    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError
