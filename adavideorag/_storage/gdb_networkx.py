import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Union
import networkx as nx

from .._utils import logger
from ..base import BaseGraphStorage


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> Optional[nx.Graph]:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        target = os.path.abspath(file_name)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(target)}.",
            suffix=".graphml",
            dir=os.path.dirname(target),
        )
        os.close(file_descriptor)
        try:
            nx.write_graphml(graph, temporary_path)
            os.replace(temporary_path, target)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        if preloaded_graph is not None and not preloaded_graph.is_directed():
            raise RuntimeError(
                "The stored graph is undirected, so temporal edge direction "
                "cannot be recovered reliably. Rebuild the working directory."
            )
        self._graph = preloaded_graph or nx.DiGraph()

    async def index_done_callback(self):
        graph_snapshot = self._graph.copy()
        await asyncio.to_thread(
            NetworkXStorage.write_nx_graph,
            graph_snapshot,
            self._graphml_xml_file,
        )

    def snapshot(self):
        return self._graph.copy()

    def restore(self, snapshot):
        if not isinstance(snapshot, nx.DiGraph):
            raise TypeError("NetworkXStorage snapshots must be directed graphs")
        self._graph = snapshot.copy()

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            if self._graph.is_directed():
                return list(
                    dict.fromkeys(
                        list(self._graph.out_edges(source_node_id))
                        + list(self._graph.in_edges(source_node_id))
                    )
                )
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)
