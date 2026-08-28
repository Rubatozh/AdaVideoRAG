import re
import json
import asyncio
import os
import tiktoken
from typing import Callable, Union
from collections import Counter, defaultdict
import io
import csv
# BENCH SEAM BEGIN
from . import bench_hooks as _bench
# BENCH SEAM END
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from ._videoutil import (
    retrieved_segment_caption_B_pipeline,
    retrieved_segment_caption_C_pipeline,
)


def _split_video_segment_id(segment_id: str) -> tuple[str, str]:
    parts = segment_id.rsplit("_", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid video segment id: {segment_id}")
    return parts[0], parts[1]


def _video_segment_sort_key(segment_id: str) -> tuple[str, int]:
    video_name, index = _split_video_segment_id(segment_id)
    return video_name, int(index)


def _chunk_segment_ids(chunk: dict) -> list[str]:
    segment_ids = chunk.get("video_segment_id", [])
    return [segment_ids] if isinstance(segment_ids, str) else list(segment_ids)


def _normalize_retrieval_query(value, fallback: str) -> Union[str, None]:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
    elif isinstance(value, (list, tuple, set)):
        normalized = " ".join(
            str(item).strip() for item in value if str(item).strip()
        )
        return normalized or None
    else:
        logger.warning(
            "Query rewrite returned unsupported value %r; using the raw query",
            value,
        )
        return fallback
    return normalized or fallback


def _is_affirmative_filter_result(result: str) -> bool:
    if not isinstance(result, str):
        return False
    answer = result.rsplit("</think>", 1)[-1]
    match = re.match(r"\s*[\"']?(yes|no)\b", answer, flags=re.IGNORECASE)
    return match is not None and match.group(1).lower() == "yes"


def _clean_record_field(value: str, *, uppercase: bool = False) -> str:
    cleaned = clean_str(value).strip()
    if (
        len(cleaned) >= 2
        and cleaned[0] in {"\"", "'"}
        and cleaned[-1] == cleaned[0]
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned.upper() if uppercase else cleaned


def _extract_json_object(text: str) -> str:
    if not isinstance(text, str):
        return text
    candidate = text.rsplit("</think>", 1)[-1].strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced is not None:
        return fenced.group(1)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        return candidate[start : end + 1]
    return candidate

def chunking_by_video_segments(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    max_token_size=1024,
):
    # make sure each segment is not larger than max_token_size
    for index in range(len(tokens_list)):
        if len(tokens_list[index]) > max_token_size:
            tokens_list[index] = tokens_list[index][:max_token_size]

    results = []
    chunk_token = []
    chunk_segment_ids = []
    chunk_order_index = 0
    for index, tokens in enumerate(tokens_list):

        if len(chunk_token) + len(tokens) <= max_token_size:
            # add new segment
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
        else:
            # save the current chunk
            chunk = tiktoken_model.decode(chunk_token)
            results.append(
                {
                    "tokens": len(chunk_token),
                    "content": chunk.strip(),
                    "chunk_order_index": chunk_order_index,
                    "video_segment_id": chunk_segment_ids,
                }
            )
            # new chunk with current segment as begin
            chunk_token = []
            chunk_segment_ids = []
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
            chunk_order_index += 1

    # save the last chunk
    if len(chunk_token) > 0:
        chunk = tiktoken_model.decode(chunk_token)
        results.append(
            {
                "tokens": len(chunk_token),
                "content": chunk.strip(),
                "chunk_order_index": chunk_order_index,
                "video_segment_id": chunk_segment_ids,
            }
        )

    return results


def chunking_by_video_segments_2(
        tokens_list: list[list[int]],
        doc_keys,
        tiktoken_model,
        max_token_size=1024,
):
    # make sure each segment is not larger than max_token_size
    for index in range(len(tokens_list)):
        if len(tokens_list[index]) > max_token_size:
            tokens_list[index] = tokens_list[index][:max_token_size]


    results = []

    for index, tokens in enumerate(tokens_list):
        chunk = tiktoken_model.decode(tokens)
        results.append(
            {
                "tokens": len(tokens),
                "content": chunk.strip(),
                "chunk_order_index": index,
                "video_segment_id": doc_keys[index],
            }
        )
    return results


def get_chunks(
    new_videos,
    chunk_type="content",
    chunk_func=chunking_by_video_segments,
    tiktoken_model_name="gpt-4o",
    **chunk_func_params,
):
    inserting_chunks = {}

    encoder = tiktoken.encoding_for_model(tiktoken_model_name)
    for video_name in sorted(new_videos):
        segment_items = sorted(
            new_videos[video_name].items(),
            key=lambda item: (
                (0, int(item[0]))
                if str(item[0]).isdigit()
                else (1, str(item[0]))
            ),
        )
        docs = []
        doc_keys = []
        for index, segment in segment_items:
            content = segment.get(chunk_type)
            if not isinstance(content, str) or not content.strip():
                logger.debug(
                    "Skipping empty %s evidence for %s_%s",
                    chunk_type,
                    video_name,
                    index,
                )
                continue
            docs.append(content)
            doc_keys.append(f"{video_name}_{index}")
        if not docs:
            continue

        tokens = encoder.encode_batch(docs, num_threads=16)
        chunks = chunk_func(
            tokens, doc_keys=doc_keys, tiktoken_model=encoder, **chunk_func_params
        )

        for chunk in chunks:
            provenance = "\0".join(_chunk_segment_ids(chunk))
            identity = f"{provenance}\0{chunk['content']}"
            inserting_chunks.update(
                {
                    compute_mdhash_id(
                        identity, prefix=f"{chunk_type}-chunk-"
                    ): chunk
                }
            )

    return inserting_chunks
async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: Callable = global_config["llm"]["cheap_model_func"]
    llm_max_tokens = global_config["llm"]["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if (
        len(record_attributes) < 4
        or _clean_record_field(record_attributes[0]).lower() != "entity"
    ):
        return None
    entity_name = _clean_record_field(record_attributes[1], uppercase=True)
    if not entity_name:
        return None
    entity_type = _clean_record_field(record_attributes[2], uppercase=True)
    if len(record_attributes) >= 5:
        temporal_attribute = _clean_record_field(record_attributes[3]) or "UNKNOWN"
        entity_description = _clean_record_field(record_attributes[4])
    else:
        # Backwards compatibility for graphs produced by the original prompt.
        temporal_attribute = "UNKNOWN"
        entity_description = _clean_record_field(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        temporal_attribute=temporal_attribute,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if (
        len(record_attributes) < 5
        or _clean_record_field(record_attributes[0]).lower() != "relationship"
    ):
        return None
    source = _clean_record_field(record_attributes[1], uppercase=True)
    target = _clean_record_field(record_attributes[2], uppercase=True)
    if not source or not target:
        return None
    if len(record_attributes) >= 6:
        relation_type = (
            _clean_record_field(record_attributes[3], uppercase=True) or "RELATED"
        )
        edge_description = _clean_record_field(record_attributes[4])
    else:
        # Backwards compatibility for graphs produced by the original prompt.
        relation_type = "RELATED"
        edge_description = _clean_record_field(record_attributes[3])

    edge_source_id = chunk_key
    raw_weight = _clean_record_field(record_attributes[-1])
    weight = (
        float(raw_weight) if is_float_regex(raw_weight) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        relation_type=relation_type,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []
    already_temporal_attributes = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
        already_temporal_attributes.extend(
            split_string_by_multi_markers(
                already_node.get("temporal_attribute", "UNKNOWN"),
                [GRAPH_FIELD_SEP],
            )
        )

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        sorted(set([dp["source_id"] for dp in nodes_data] + already_source_ids))
    )
    temporal_attributes = set(
        [dp.get("temporal_attribute", "UNKNOWN") for dp in nodes_data]
        + already_temporal_attributes
    )
    if len(temporal_attributes) > 1:
        temporal_attributes.discard("UNKNOWN")
    temporal_attribute = GRAPH_FIELD_SEP.join(sorted(temporal_attributes))
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        temporal_attribute=temporal_attribute,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    already_relation_types = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))
        already_relation_types.extend(
            split_string_by_multi_markers(
                already_edge.get("relation_type", "RELATED"),
                [GRAPH_FIELD_SEP],
            )
        )


    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    # Treat one source chunk as one unit of evidence. This makes graph updates
    # idempotent when a process stops after the graph is saved but before the
    # text-chunk completion marker is committed, then retries the same video.
    weight_by_source = {}
    for edge_data in edges_data:
        source = edge_data["source_id"]
        weight_by_source[source] = max(
            edge_data["weight"],
            weight_by_source.get(source, float("-inf")),
        )
    new_weight = sum(
        edge_weight
        for source, edge_weight in weight_by_source.items()
        if source not in already_source_ids
    )
    weight = sum(already_weights) + new_weight
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        sorted(set([dp["source_id"] for dp in edges_data] + already_source_ids))
    )
    relation_types = set(
        [dp.get("relation_type", "RELATED") for dp in edges_data]
        + already_relation_types
    )
    if len(relation_types) > 1:
        relation_types.discard("RELATED")
    relation_type = GRAPH_FIELD_SEP.join(sorted(relation_types))
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": "UNKNOWN",
                    "temporal_attribute": "UNKNOWN",
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            relation_type=relation_type,
            description=description,
            source_id=source_id,
            order=order,
        ),
    )
    return_edge_data = dict(
        src_tgt=(src_id, tgt_id),
        description=description,
        relation_type=relation_type,
        weight=weight,
    )
    return return_edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationship_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:

    use_llm_func: Callable = global_config["llm"]["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entity_continue_extraction"]
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        logger.debug(
            "Processed %d chunks, %d entities (including duplicates), "
            "%d relations (including duplicates)",
            already_processed,
            already_entities,
            already_relations,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # Preserve source/target direction for temporal and causal edges.
            maybe_edges[k].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_edges_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not all_entities_data and not all_edges_data:
        logger.warning("No entities or relationships were extracted from the chunks")
        return None

    if entity_vdb is not None and all_entities_data:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": (
                    f"{dp['entity_name']}\nTime: {dp['temporal_attribute']}\n"
                    f"{dp['description']}"
                ),
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationship_vdb is not None and all_edges_data:
        data_for_vdb = {
            compute_mdhash_id(
                f"{dp['src_tgt'][0]}\0{dp['src_tgt'][1]}", prefix="rel-"
            ): {
                "content": (
                    f"{dp['src_tgt'][0]}\t{dp['src_tgt'][1]}\n"
                    f"Type: {dp['relation_type']}\n{dp['description']}"
                ),
                "src_id": dp["src_tgt"][0],
                "tgt_id": dp["src_tgt"][1],
            }
            for dp in all_edges_data
        }
        await relationship_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst, all_entities_data, all_edges_data


async def _find_most_related_segments_from_entities(
    topk_chunks: int,
    node_datas: list[dict],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for node_data, this_edges in zip(node_datas, edges):
        if not this_edges:
            continue
        entity_name = node_data["entity_name"]
        all_one_hop_nodes.update(
            e[1] if e[0] == entity_name else e[0] for e in this_edges
        )
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges or []:
                neighbor = e[1] if e[0] == node_datas[index]["entity_name"] else e[0]
                if (
                    neighbor in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[neighbor]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any(v["data"] is None for v in all_text_units_lookup.values()):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v["data"] is not None
    ]
    sorted_text_units = sorted(
        all_text_units, key=lambda x: -x["relation_counts"]
    )[:topk_chunks]



    chunk_related_segments = set()
    for _chunk_data in sorted_text_units:
        for s_id in _chunk_segment_ids(_chunk_data["data"]):
            chunk_related_segments.add(s_id)

    return chunk_related_segments

async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        if not this_edges:
            continue
        for e in this_edges:
            directed_edge = tuple(e)
            if directed_edge not in seen:
                seen.add(directed_edge)
                all_edges.append(directed_edge)

    all_edges_pack, all_edges_degree = await asyncio.gather(
        asyncio.gather(*[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]),
        asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
        ),
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )


    return all_edges_data


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[
                knowledge_graph_inst.get_node(entity_name)
                for entity_name in entity_names
            ]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.node_degree(entity_name)
                for entity_name in entity_names
            ]
        ),
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
        if n is not None
    ]


    return node_datas


async def _find_most_related_segments_from_relationships(
    topk_chunks: int,
    edge_datas: list[dict],
    text_chunks_db: BaseKVStorage,
):

    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    chunk_orders = {}
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            chunk_orders[c_id] = min(index, chunk_orders.get(c_id, index))

    chunk_ids = list(chunk_orders)
    chunk_values = await text_chunks_db.get_by_ids(chunk_ids)
    all_text_units_lookup = {
        chunk_id: {
            "data": chunk_data,
            "order": chunk_orders[chunk_id],
        }
        for chunk_id, chunk_data in zip(chunk_ids, chunk_values)
        if chunk_data is not None and "content" in chunk_data
    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []


    chunk_related_segments = set()
    for _chunk_data in valid_text_units[:topk_chunks]:
        for s_id in _chunk_segment_ids(_chunk_data["data"]):
            chunk_related_segments.add(s_id)

    return chunk_related_segments



async def _refine_entity_retrieval_query(
    query,
    global_config: dict,
):
    use_llm_func: Callable = global_config["llm"]["cheap_model_func"]
    query_rewrite_prompt = PROMPTS["query_rewrite_for_entity_retrieval"]
    query_rewrite_prompt = query_rewrite_prompt.format(input_text=query)
    final_result = await use_llm_func(query_rewrite_prompt)
    if "</think>" in final_result:
        final_result = final_result.split('</think>')[1].strip()
    return final_result.strip() or query


async def _refine_query(
    query,
    global_config: dict,
):
    use_llm_func: Callable = global_config["llm"]["cheap_model_func"]
    query_rewrite_prompt = PROMPTS["query_rewrite"]
    query_input = PROMPTS["input"].format(input_text=query)
    query_rewrite_prompt = query_rewrite_prompt + query_input
    final_result = await use_llm_func(query_rewrite_prompt)
    return _extract_json_object(final_result)

async def _refine_visual_retrieval_query(
    query,
    global_config: dict,
):
    use_llm_func: Callable = global_config["llm"]["cheap_model_func"]
    query_rewrite_prompt = PROMPTS["query_rewrite_for_visual_retrieval"]
    query_rewrite_prompt = query_rewrite_prompt.format(input_text=query)
    final_result = await use_llm_func(query_rewrite_prompt)
    if "</think>" in final_result:
        final_result = final_result.split('</think>')[1].strip()
    return final_result.strip() or query
async def videorag_query_B_pipeline(
        query,
        caption_vdb,
        ASR_vdb,
        ocr_vdb,
        video_path_db,
        video_segments,
        video_segment_feature_vdb,
        vlm_model,
        query_param: QueryParam,
        global_config: dict,
) -> str:

    use_model_func = global_config["llm"]["best_model_func"]
    refined_query = await _refine_query(query, global_config)
    try:
        retrieval_queries = json.loads(refined_query)
        if not isinstance(retrieval_queries, dict):
            raise TypeError("query rewrite must be a JSON object")
        caption_query = _normalize_retrieval_query(
            retrieval_queries.get("caption", query), query
        )
        ASR_query = _normalize_retrieval_query(
            retrieval_queries.get("ASR", query), query
        )
        OCR_query = _normalize_retrieval_query(
            retrieval_queries.get("OCR", query), query
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.warning("Query rewrite did not return valid JSON; using raw query")
        caption_query = query
        ASR_query = query
        OCR_query = query

    text_segments = set()
    caption_retrieved_segments = set()
    if caption_query is not None:
        caption_results = await caption_vdb.query(caption_query, top_k=query_param.top_k)
        if len(caption_results):
            for n in caption_results:
                caption_retrieved_segments.add(n['__video_id__'])
        text_segments = text_segments.union(caption_retrieved_segments)
    if ASR_query is not None:
        ASR_results = await ASR_vdb.query(ASR_query, top_k=query_param.top_k)
        ASR_retrieved_segments = set()
        if len(ASR_results):
            for n in ASR_results:
                ASR_retrieved_segments.add(n['__video_id__'])
        text_segments = text_segments.union(ASR_retrieved_segments)
    if OCR_query is not None:
        ocr_results = await ocr_vdb.query(OCR_query, top_k=query_param.top_k)
        ocr_retrieved_segments = set()
        if len(ocr_results):
            for n in ocr_results:
                ocr_retrieved_segments.add(n['__video_id__'])
        text_segments = text_segments.union(ocr_retrieved_segments)


    # visual retrieval
    query_for_visual_retrieval = caption_query or query

    segment_results = await video_segment_feature_vdb.query(query_for_visual_retrieval)
    visual_retrieved_segments = set()
    if len(segment_results):
        for n in segment_results:
            visual_retrieved_segments.add(n['__id__'])


    retrieved_segments = list(text_segments.union(visual_retrieved_segments))
    logger.info("Retrieved text segments: %s", sorted(text_segments))

    retrieved_segments = sorted(
        retrieved_segments,
        key=_video_segment_sort_key,
    )
# BENCH SEAM BEGIN
    retrieved_segments = _bench.retrieved(retrieved_segments, query=query)
# BENCH SEAM END

    logger.debug("Visual retrieval query: %s", query_for_visual_retrieval)
    logger.info("Retrieved visual segments: %s", sorted(visual_retrieved_segments))

    already_processed = 0

    async def _filter_single_segment(knowledge: str, segment_key_dp: tuple[str, str]):
        nonlocal use_model_func, already_processed
        segment_key = segment_key_dp[0]
        segment_content = segment_key_dp[1]
        filter_prompt = PROMPTS["filtering_segment"]
        filter_prompt = filter_prompt.format(caption=segment_content, knowledge=knowledge)
        result = await use_model_func(filter_prompt)
        already_processed += 1
        logger.debug("Checked %d segments", already_processed)
        return (segment_key, result)
    rough_captions = {}
    for this_segment in retrieved_segments:
        video_name, index = _split_video_segment_id(this_segment)
        segment = video_segments.data.get(video_name, {}).get(index)
        if segment is not None:
            rough_captions[this_segment] = segment["content"]
    retrieved_segments = list(rough_captions)

    if not video_path_db.data:
        logger.warning("No indexed video is available for generation")
        return PROMPTS["fail_response"]
    if not retrieved_segments:
        logger.warning("No valid video segment was retrieved for Level B generation")
        return PROMPTS["fail_response"]

    results = await asyncio.gather(
        *[_filter_single_segment(query, (s_id, rough_captions[s_id])) for s_id in
          rough_captions]
    )

    remain_segments = [x[0] for x in results if _is_affirmative_filter_result(x[1])]
# BENCH SEAM BEGIN
    remain_segments = _bench.filtered(remain_segments, query=query)
# BENCH SEAM END



    logger.info("%d video segments remain after filtering", len(remain_segments))
    if len(remain_segments) == 0:
        logger.info("No segment passed filtering; using all retrieved segments")
        remain_segments = retrieved_segments
    logger.debug("Segments used for generation: %s", remain_segments)




    response = await retrieved_segment_caption_B_pipeline(
        query,
        vlm_model,
        remain_segments,
        video_path_db,
        video_segments,
        work_dir = global_config['working_dir'],
    )


    return response



async def videorag_query_C_pipeline(
        query,
        entities_vdb,
        relationship_vdb,
        text_chunks_db,
        chunks_vdb,
        video_path_db,
        video_segments,
        video_segment_feature_vdb,
        knowledge_graph_inst,
        vlm_model,
        query_param: QueryParam,
        global_config: dict,
) -> str:
    use_model_func = global_config["llm"]["best_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)

    chunks_ids = [r["id"] for r in results]
    chunks = [
        chunk
        for chunk in await text_chunks_db.get_by_ids(chunks_ids)
        if chunk is not None
    ]
    truncated_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
        model_name=global_config["tiktoken_model_name"],
    )
    logger.info(f"Truncate {len(chunks)} to {len(truncated_chunks)} chunks")
    retrieved_chunk_context = "-----New Chunk-----\n".join(
        chunk["content"] for chunk in truncated_chunks
    )


    query_for_entity_retrieval = await _refine_entity_retrieval_query(
        query,
        global_config,
    )

    entity_data, edge_data = await asyncio.gather(
        _get_node_data(
            query_for_entity_retrieval,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
            global_config
        ),
        _get_edge_data(
            query_for_entity_retrieval,
            knowledge_graph_inst,
            relationship_vdb,
            text_chunks_db,
            query_param,
            global_config
        ),
    )

    (
        entity_retrieved_segments,
        ll_entities_context,
        ll_relations_context,
    ) = entity_data

    if edge_data is not None:
        (
            relationship_retrieved_segments,
            hl_entities_context,
            hl_relations_context,
        ) = edge_data

        entities_context, relations_context= combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
        )
    else:
        entities_context, relations_context =ll_entities_context, ll_relations_context



    # visual retrieval
    query_for_visual_retrieval = await _refine_visual_retrieval_query(
        query,
        global_config,
    )
    segment_results = await video_segment_feature_vdb.query(query_for_visual_retrieval)
    visual_retrieved_segments = set()
    if len(segment_results):
        for n in segment_results:
            visual_retrieved_segments.add(n['__id__'])

    if edge_data is not None:
        retrieved_segments = list(entity_retrieved_segments.union(relationship_retrieved_segments).union(visual_retrieved_segments))
    else:
        retrieved_segments = list(entity_retrieved_segments.union(visual_retrieved_segments))
    logger.info("Retrieved graph segments: %s", sorted(entity_retrieved_segments))

    retrieved_segments = sorted(
        retrieved_segments,
        key=_video_segment_sort_key,
    )
# BENCH SEAM BEGIN
    retrieved_segments = _bench.retrieved(retrieved_segments, query=query)
# BENCH SEAM END

    logger.debug("Visual retrieval query: %s", query_for_visual_retrieval)
    logger.info("Retrieved visual segments: %s", sorted(visual_retrieved_segments))

    already_processed = 0

    async def _filter_single_segment(knowledge: str, segment_key_dp: tuple[str, str]):
        nonlocal use_model_func, already_processed
        segment_key = segment_key_dp[0]
        segment_content = segment_key_dp[1]
        filter_prompt = PROMPTS["filtering_segment"]
        filter_prompt = filter_prompt.format(caption=segment_content, knowledge=knowledge)
        result = await use_model_func(filter_prompt)
        already_processed += 1
        logger.debug("Checked %d segments", already_processed)
        return (segment_key, result)

    rough_captions = {}
    for s_id in retrieved_segments:
        video_name, index = _split_video_segment_id(s_id)
        segment = video_segments.data.get(video_name, {}).get(index)
        if segment is not None:
            rough_captions[s_id] = segment["content"]
    retrieved_segments = list(rough_captions)

    if not video_path_db.data:
        logger.warning("No indexed video is available for generation")
        return PROMPTS["fail_response"]
    if not retrieved_segments:
        logger.warning("No valid video segment was retrieved for Level C generation")
        return PROMPTS["fail_response"]
    results = await asyncio.gather(
        *[_filter_single_segment(query, (s_id, rough_captions[s_id])) for s_id in rough_captions]
    )
    remain_segments = [x[0] for x in results if _is_affirmative_filter_result(x[1])]
# BENCH SEAM BEGIN
    remain_segments = _bench.filtered(remain_segments, query=query)
# BENCH SEAM END
    logger.info("%d video segments remain after filtering", len(remain_segments))

    if len(remain_segments) == 0:
        logger.info("No segment passed filtering; using all retrieved segments")
        remain_segments = retrieved_segments
    logger.debug("Segments used for generation: %s", remain_segments)

    response = await retrieved_segment_caption_C_pipeline(
        query,
        vlm_model,
        retrieved_chunk_context,
        remain_segments,
        video_path_db,
        video_segments,
        entities_context,
        relations_context,
        work_dir = global_config['working_dir'],
    )


    return response






async def videorag_query_A(
        query,
        image_dict,
        video_dict,
        vlm_model,
) -> str:
    query = "Answer the question. Question: " + query
    media_paths = [
        path
        for path in [*video_dict.data.values(), *image_dict.data.values()]
        if path and os.path.isfile(path)
    ]
    if media_paths:
        media_input = media_paths[0] if len(media_paths) == 1 else media_paths
        return await vlm_model.process_video(media_input, query)
    logger.warning("No indexed video or image is available for Level A generation")
    return PROMPTS["fail_response"]



async def _get_node_data(query_for_entity_retrieval, knowledge_graph_inst, entities_vdb, text_chunks_db,  query_param, global_config):
    entity_results = await entities_vdb.query(query_for_entity_retrieval, top_k=query_param.top_k)
    entity_retrieved_segments = set()
    use_relations = set()
    node_datas = []
    if len(entity_results):
        node_datas = await asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_results]
        )
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
        node_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_results]
        )
        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(entity_results, node_datas, node_degrees)
            if n is not None
        ]
        entity_retrieved_segments, use_relations = await asyncio.gather(
            _find_most_related_segments_from_entities(global_config["retrieval_topk_chunks"], node_datas,text_chunks_db, knowledge_graph_inst),
            _find_most_related_edges_from_entities(node_datas, knowledge_graph_inst),
        )

    # build prompt
    entites_section_list = [
        [
            "id",
            "entity",
            "type",
            "time",
            "description",
            "rank",
        ]
    ]
    for i, n in enumerate(node_datas):

        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("temporal_attribute", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "relation_type",
            "description",
            "weight",
            "rank",

        ]
    ]
    for i, e in enumerate(use_relations):

        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e.get("relation_type", "RELATED"),
                e["description"],
                e["weight"],
                e["rank"],

            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    return  entity_retrieved_segments, entities_context, relations_context


async def _get_edge_data(query_for_entity_retrieval, knowledge_graph_inst, relationship_vdb, text_chunks_db, query_param, global_config):
    relationship_results = await  relationship_vdb.query(query_for_entity_retrieval, top_k=query_param.top_k)

    if len(relationship_results):
        edge_datas, edge_degree = await asyncio.gather(
            asyncio.gather(
                *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in relationship_results]
            ),
            asyncio.gather(
                *[
                    knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"])
                    for r in relationship_results
                ]
            ),
        )
        edge_datas = [
            {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "rank": d,
                **v,
            }
            for k, v, d in zip(relationship_results, edge_datas, edge_degree)
            if v is not None
        ]

        edge_datas = sorted(
            edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )

        relationship_retrieved_segments, use_entities = await asyncio.gather(

                _find_most_related_segments_from_relationships(
                    global_config["retrieval_topk_chunks"],
                    edge_datas,
                    text_chunks_db,
                ),
               _find_most_related_entities_from_relationships(edge_datas, knowledge_graph_inst),
        )

        relations_section_list = [
            [
                "id",
                "source",
                "target",
                "relation_type",
                "description",
                "weight",
                "rank",

            ]
        ]
        for i, e in enumerate(edge_datas):

            relations_section_list.append(
                [
                    i,
                    e["src_id"],
                    e["tgt_id"],
                    e.get("relation_type", "RELATED"),
                    e["description"],
                    e["weight"],
                    e["rank"],

                ]
            )
        relations_context = list_of_list_to_csv(relations_section_list)

        entites_section_list = [
            ["id", "entity", "type", "time", "description", "rank"]
        ]
        for i, n in enumerate(use_entities):
            entites_section_list.append(
                [
                    i,
                    n["entity_name"],
                    n.get("entity_type", "UNKNOWN"),
                    n.get("temporal_attribute", "UNKNOWN"),
                    n.get("description", "UNKNOWN"),
                    n["rank"],

                ]
            )
        entities_context = list_of_list_to_csv(entites_section_list)

        return relationship_retrieved_segments, entities_context, relations_context
    return None


def combine_contexts(entities, relationships):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]

    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    return combined_entities, combined_relationships


def process_combine_contexts(hl: str, ll: str):
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    combined_rows = []
    seen = set()

    for row in list_hl + list_ll:
        if not row:
            continue
        values = tuple(row[1:])
        if values not in seen:
            seen.add(values)
            combined_rows.append(values)

    return list_of_list_to_csv(
        [header]
        + [[index, *values] for index, values in enumerate(combined_rows, start=1)]
    )


def csv_string_to_list(csv_string: str) -> list[list[str]]:
    # Clean the string by removing NUL characters
    cleaned_string = csv_string.replace("\0", "")

    output = io.StringIO(cleaned_string)
    reader = csv.reader(
        output,
        quoting=csv.QUOTE_ALL,  # Match the writer configuration
        escapechar="\\",  # Use backslash as escape character
        quotechar='"',  # Use double quotes
    )

    try:
        return [row for row in reader]
    except csv.Error as e:
        raise ValueError(f"Failed to parse CSV string: {str(e)}")
    finally:
        output.close()
