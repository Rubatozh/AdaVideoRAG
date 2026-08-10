import os
import asyncio
import copy
import shutil
from dataclasses import dataclass, field, fields
from datetime import datetime
from functools import partial
from hashlib import sha256
from typing import Callable, Dict, List, Optional, Type, Union, cast

import tiktoken
from transformers import AutoModel, AutoTokenizer

from ._llm import (
    LLMConfig,
    openai_config,
)
from ._op import (
    chunking_by_video_segments,
    chunking_by_video_segments_2,
    extract_entities,
    get_chunks,
    videorag_query_A,
    videorag_query_B_pipeline,
    videorag_query_C_pipeline,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NanoVectorDBAndVideoStorage,
    NanoVectorDBVideoSegmentStorage,
    NetworkXStorage,
)
from ._utils import (
    limit_async_func_call,
    load_json,
    wrap_embedding_func_with_attrs,
    logger,
    write_json,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from ._videoutil import deal_video


IMAGE_EXTENSIONS = {"gif", "jpeg", "jpg", "png", "webp"}
VIDEO_EXTENSIONS = {"avi", "m4v", "mkv", "mov", "mp4", "webm"}
INDEX_SCHEMA_VERSION = 4


def _ensure_index_schema(working_dir: str, index_config: dict) -> None:
    manifest_path = os.path.join(working_dir, "index_schema.json")
    manifest = load_json(manifest_path)
    expected_manifest = {
        "version": INDEX_SCHEMA_VERSION,
        "index_config": index_config,
    }
    if manifest is None:
        legacy_indexes = [
            name
            for name in os.listdir(working_dir)
            if name.startswith(("graph_", "kv_store_", "vdb_"))
        ]
        if legacy_indexes:
            raise RuntimeError(
                "The working directory contains an index without a compatible "
                "schema manifest. Rebuild it in a new --working-dir."
            )
        write_json(expected_manifest, manifest_path)
        return

    if not isinstance(manifest, dict) or manifest.get("version") != INDEX_SCHEMA_VERSION:
        found_version = manifest.get("version") if isinstance(manifest, dict) else None
        raise RuntimeError(
            "Incompatible index schema: "
            f"expected {INDEX_SCHEMA_VERSION}, found {found_version!r}. "
            "Rebuild the index in a new --working-dir."
        )
    if manifest.get("index_config") != index_config:
        raise RuntimeError(
            "The indexing configuration differs from the configuration stored "
            "in this working directory. Rebuild the index in a new --working-dir."
        )


def _media_key(media_path: str, existing_paths) -> str:
    """Keep readable IDs while disambiguating equal basenames."""
    normalized_path = os.path.realpath(os.path.expanduser(media_path))
    stem = os.path.splitext(os.path.basename(normalized_path))[0]
    if len(stem.encode("utf-8")) > 96:
        shortened = stem.encode("utf-8")[:80].decode("utf-8", errors="ignore")
        stem = f"{shortened}-{sha256(normalized_path.encode()).hexdigest()[:12]}"
    existing_path = existing_paths.get(stem)
    if existing_path is None or os.path.realpath(existing_path) == normalized_path:
        return stem

    digest = sha256(normalized_path.encode("utf-8")).hexdigest()
    for suffix_length in range(8, len(digest) + 1, 4):
        candidate = f"{stem}-{digest[:suffix_length]}"
        candidate_path = existing_paths.get(candidate)
        if (
            candidate_path is None
            or os.path.realpath(candidate_path) == normalized_path
        ):
            return candidate
    raise RuntimeError(f"Unable to create a unique media key for {media_path}")


@dataclass
class AdaVideoRAG:

    working_dir: str = field(
        default_factory=lambda: (
            f"./adavideorag_cache_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        )
    )

    cuda: str = None

    # video
    video_segment_length: int = 30  # seconds
    rough_num_frames_per_segment: int = 30  # frames
    fine_num_frames_per_segment: int = 15  # frames
    video_output_format: str = "mp4"
    audio_output_format: str = "mp3"
    video_embedding_batch_num: int = 2
    segment_retrieval_top_k: int = 4
    video_embedding_dim: int = 1024



    # query
    retrieval_topk_chunks: int = 2
    query_better_than_threshold: float = 0.2
    visual_query_better_than_threshold: float = 0.5


    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_video_segments

    vector_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_video_segments_2

    chunk_token_size: int = 1200
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # Change to your LLM provider
    llm: LLMConfig = field(default_factory=lambda: copy.deepcopy(openai_config))

    entity_extraction_func: Callable = extract_entities
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_2: Type[BaseVectorStorage] = NanoVectorDBAndVideoStorage
    vs_vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBVideoSegmentStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    level: str = "Level A"
    caption_model: Optional[Callable] = None
    caption_tokenizer: Optional[Callable] = None
    caption_model_path: str = "./MiniCPM-V-2_6-int4"
    whisper_model_path: str = "./faster-distil-whisper-large-v3"
    ocr_languages: tuple[str, ...] = ("en",)

    def load_caption_model(self):
        if self.caption_model is not None and self.caption_tokenizer is not None:
            return

        if not os.path.isdir(self.caption_model_path):
            raise FileNotFoundError(
                f"Caption model not found: {self.caption_model_path}. "
                "Download MiniCPM-V-2_6-int4 or pass caption_model_path."
            )

        self.caption_model = AutoModel.from_pretrained(
            self.caption_model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.caption_tokenizer = AutoTokenizer.from_pretrained(
            self.caption_model_path,
            trust_remote_code=True,
        )
        self.caption_model.eval()

    def set_vlm_model(self, vlm_model):
        if not callable(getattr(vlm_model, "process_video", None)):
            raise TypeError("vlm_model must provide an async process_video method")
        self.vlm_model = vlm_model

    def _global_config(self):
        """Build a lightweight runtime config without copying loaded models."""
        config = {
            config_field.name: getattr(self, config_field.name)
            for config_field in fields(self)
        }
        llm_config = {
            config_field.name: getattr(self.llm, config_field.name)
            for config_field in fields(self.llm)
        }
        # Entity extraction and query code must use the cache/concurrency-aware
        # wrappers created on this AdaVideoRAG instance.
        llm_config["best_model_func"] = getattr(
            self, "best_model_func", self.llm.best_model_func
        )
        llm_config["cheap_model_func"] = getattr(
            self, "cheap_model_func", self.llm.cheap_model_func
        )
        config["llm"] = llm_config
        # Runtime model objects are passed directly where needed; storages and
        # operators should never deep-copy or serialize them.
        config["caption_model"] = None
        config["caption_tokenizer"] = None
        return config

    def __post_init__(self):
        logger.debug(
            "AdaVideoRAG init: working_dir=%s, level=%s",
            self.working_dir,
            self.level,
        )


        if not self.working_dir:
            raise ValueError("working_dir must be a non-empty path")
        if self.level not in {"Level A", "Level B", "Level C"}:
            raise ValueError(f"Unknown query level: {self.level}")
        if self.video_segment_length <= 0:
            raise ValueError("video_segment_length must be positive")
        if self.rough_num_frames_per_segment <= 0:
            raise ValueError("rough_num_frames_per_segment must be positive")
        if self.fine_num_frames_per_segment <= 0:
            raise ValueError("fine_num_frames_per_segment must be positive")
        if self.chunk_token_size <= 0:
            raise ValueError("chunk_token_size must be positive")
        if self.segment_retrieval_top_k <= 0:
            raise ValueError("segment_retrieval_top_k must be positive")
        if self.retrieval_topk_chunks <= 0:
            raise ValueError("retrieval_topk_chunks must be positive")
        if self.video_embedding_batch_num <= 0:
            raise ValueError("video_embedding_batch_num must be positive")
        if self.video_embedding_dim <= 0:
            raise ValueError("video_embedding_dim must be positive")
        if self.llm.embedding_batch_num <= 0:
            raise ValueError("llm.embedding_batch_num must be positive")
        if self.llm.embedding_func_max_async <= 0:
            raise ValueError("llm.embedding_func_max_async must be positive")
        if self.llm.best_model_max_async <= 0:
            raise ValueError("llm.best_model_max_async must be positive")
        if self.llm.cheap_model_max_async <= 0:
            raise ValueError("llm.cheap_model_max_async must be positive")
        for name, threshold in (
            ("query_better_than_threshold", self.query_better_than_threshold),
            (
                "visual_query_better_than_threshold",
                self.visual_query_better_than_threshold,
            ),
        ):
            if not -1.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be between -1 and 1")
        if isinstance(self.ocr_languages, str):
            self.ocr_languages = tuple(
                language.strip()
                for language in self.ocr_languages.split(",")
                if language.strip()
            )
        else:
            self.ocr_languages = tuple(self.ocr_languages)
        if not self.ocr_languages:
            raise ValueError("ocr_languages must contain at least one language")

        self.working_dir = os.path.abspath(os.path.expanduser(self.working_dir))
        self.caption_model_path = os.path.abspath(
            os.path.expanduser(self.caption_model_path)
        )
        self.whisper_model_path = os.path.expanduser(self.whisper_model_path)
        if os.path.isabs(self.whisper_model_path) or self.whisper_model_path.startswith(
            "."
        ):
            self.whisper_model_path = os.path.abspath(self.whisper_model_path)

        if not os.path.exists(self.working_dir):
            if not self.always_create_working_dir:
                raise FileNotFoundError(
                    f"Working directory does not exist: {self.working_dir}"
                )
            logger.info("Creating working directory %s", self.working_dir)
            os.makedirs(self.working_dir, exist_ok=True)

        _ensure_index_schema(
            self.working_dir,
            {
                "audio_output_format": self.audio_output_format,
                "best_model": self.llm.best_model_name,
                "caption_model": self.caption_model_path,
                "chunk_token_size": self.chunk_token_size,
                "cheap_model": self.llm.cheap_model_name,
                "embedding_dim": self.llm.embedding_dim,
                "embedding_model": self.llm.embedding_model_name,
                "fine_num_frames_per_segment": self.fine_num_frames_per_segment,
                "ocr_languages": list(self.ocr_languages),
                "rough_num_frames_per_segment": self.rough_num_frames_per_segment,
                "segment_length": self.video_segment_length,
                "tiktoken_model": self.tiktoken_model_name,
                "video_embedding_dim": self.video_embedding_dim,
                "video_output_format": self.video_output_format,
                "whisper_model": self.whisper_model_path,
            },
        )
        self._insert_lock = asyncio.Lock()
        self._video_preprocessor_cache = {}

        self.image_dict = self.key_string_value_json_storage_cls(
            namespace="image_dict", global_config=self._global_config()
        )
        self.video_dict = self.key_string_value_json_storage_cls(
            namespace="video_dict", global_config=self._global_config()
        )
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=self._global_config()
            )
            if self.enable_llm_cache
            else None
        )

        self.embedding_func = limit_async_func_call(self.llm.embedding_func_max_async)(
            wrap_embedding_func_with_attrs(
                embedding_dim=self.llm.embedding_dim,
                max_token_size=self.llm.embedding_max_token_size,
                model_name=self.llm.embedding_model_name)(self.llm.embedding_func))

        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=self._global_config())
        self.entities_vdb = self.vector_db_storage_cls(
                namespace="entities",
                global_config=self._global_config(),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=self._global_config(),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )





        self.best_model_func = limit_async_func_call(self.llm.best_model_max_async)(
            partial(self.llm.best_model_func, hashing_kv=self.llm_response_cache)
        )

        self.cheap_model_func = limit_async_func_call(self.llm.cheap_model_max_async)(
            partial(self.llm.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=self._global_config()

        )
        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=self._global_config()

        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=self._global_config()
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=self._global_config(),
            embedding_func=self.embedding_func,
        )
        self.caption_vdb = self.vector_db_storage_cls_2(
            namespace="captions",
            global_config=self._global_config(),
            embedding_func=self.embedding_func,
            cosine_better_than_threshold=0.3
        )

        self.ASR_vdb = self.vector_db_storage_cls_2(
            namespace="ASR",
            global_config=self._global_config(),
            embedding_func=self.embedding_func,
            cosine_better_than_threshold=0.3
        )

        self.OCR_vdb = self.vector_db_storage_cls_2(
            namespace="ocrs",
            global_config=self._global_config(),
            embedding_func=self.embedding_func,
            cosine_better_than_threshold=0.3
        )

        self.video_segment_feature_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_segment_feature",
                global_config=self._global_config(),
                embedding_func=None,  # we code the embedding process inside the insert() function.
            )
        )

    async def insert_video(self, video_path_list=None):
        async with self._insert_lock:
            await self._insert_video(video_path_list)

    async def _insert_video(self, video_path_list=None):
        if not video_path_list:
            raise ValueError("video_path_list must contain at least one input path")

        video_path_list = [
            os.path.realpath(os.path.expanduser(path)) for path in video_path_list
        ]
        for video_path in video_path_list:
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Input media does not exist: {video_path}")

        if self.level == 'Level A':
            for video_path in video_path_list:
                existing_media = {**self.image_dict.data, **self.video_dict.data}
                video_name = _media_key(video_path, existing_media)
                _, extension = os.path.splitext(video_path)
                extension = extension.lstrip('.').lower()
                if video_name in self.image_dict.data or video_name in self.video_dict.data:
                    logger.info("Media already indexed; skipping %s", video_path)
                    continue
                if extension in VIDEO_EXTENSIONS:
                    await self.video_dict.upsert({video_name: video_path})
                elif extension in IMAGE_EXTENSIONS:
                    await self.image_dict.upsert({video_name: video_path})
                else:
                    raise ValueError(f"Unsupported media extension: {extension}")
            await self._save_level_a_media()
        else:
            new_video_segments = {}
            new_video_paths = {}
            known_video_paths = dict(self.video_path_db.data)
            for video_path in video_path_list:
                video_name = _media_key(video_path, known_video_paths)
                _, extension = os.path.splitext(video_path)
                extension = extension.lstrip('.').lower()

                if (
                    video_name in self.video_segments.data
                    or video_name in new_video_segments
                ):
                    logger.info("Video already indexed; skipping %s", video_path)
                    continue

                if extension not in VIDEO_EXTENSIONS:
                    raise ValueError(
                        f"Level B/C indexing requires a video file, got: {video_path}"
                    )

                if self.caption_model is None or self.caption_tokenizer is None:
                    await asyncio.to_thread(self.load_caption_model)

                video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)

                if os.path.exists(video_segment_cache_path):
                    await self._cleanup_segment_cache(video_segment_cache_path)
                os.makedirs(video_segment_cache_path, exist_ok=False)

                try:
                    segments_information = await self._process_single_video(
                        video_path, video_segment_cache_path, video_name
                    )
                    new_video_segments[video_name] = segments_information
                    new_video_paths[video_name] = video_path
                    known_video_paths[video_name] = video_path
                finally:
                    await self._cleanup_segment_cache(video_segment_cache_path)

            if new_video_segments:
                await self.ainsert(new_video_segments)
                await self.video_path_db.upsert(new_video_paths)
                await self.video_dict.upsert(new_video_paths)
                await self._commit_storages(self.video_path_db, self.video_dict)

                # video_segments is the durable completion marker. Persist it
                # only after all derived indexes and source paths are saved, so
                # interrupted videos remain retryable on the next run.
                await self.video_segments.upsert(new_video_segments)
                await self._commit_storages(self.video_segments)

    async def _cleanup_segment_cache(self, video_segment_cache_path):
        if os.path.exists(video_segment_cache_path):
            await asyncio.to_thread(shutil.rmtree, video_segment_cache_path)

    async def _process_single_video(self, video_path, video_segment_cache_path, video_name):
        ## split video and get caption\asr\ocrs

        loop = asyncio.get_running_loop()
        segment_index2name, segments_information = await loop.run_in_executor(
            None,
            deal_video,
            video_path,
            video_segment_cache_path,
            video_name,
            self.video_segment_length,
            self.rough_num_frames_per_segment,
            self.fine_num_frames_per_segment,
            self.audio_output_format,
            self.caption_model,
            self.caption_tokenizer,
            self.video_output_format,
            self.whisper_model_path,
            self.ocr_languages,
            self._video_preprocessor_cache,
        )
        ### save video vector
        await self.video_segment_feature_vdb.upsert(video_name, segment_index2name, self.video_output_format)
        return segments_information

    def set_level(self, level: str):
        if level not in {"Level A", "Level B", "Level C"}:
            raise ValueError(f"Unknown query level: {level}")
        self.level = level


    def query(self, query: str, param: QueryParam = None):
        param = param or QueryParam(mode="adavideorag")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aquery(query, param))
        raise RuntimeError("query() cannot run inside an event loop; await aquery()")

    async def aquery(self, query: str, param: QueryParam = None):
        param = param or QueryParam(mode="adavideorag")
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not hasattr(self, "vlm_model") or self.vlm_model is None:
            raise RuntimeError("VLM model is not configured; call set_vlm_model() first")
        try:
            if param.mode != "adavideorag":
                raise ValueError(f"Unknown mode {param.mode}")
            if self.level == 'Level A':
                response = await videorag_query_A(
                    query,
                    self.image_dict,
                    self.video_dict,
                    self.vlm_model
                )
            elif self.level =='Level B':
                    response = await videorag_query_B_pipeline(
                        query,
                        self.caption_vdb,
                        self.ASR_vdb,
                        self.OCR_vdb,
                        self.video_path_db,
                        self.video_segments,
                        self.video_segment_feature_vdb,
                        self.vlm_model,
                        param,
                        self._global_config(),
                    )
            elif self.level =='Level C':
                    response = await videorag_query_C_pipeline(
                        query,
                        self.entities_vdb,
                        self.relationships_vdb,
                        self.text_chunks,
                        self.chunks_vdb,
                        self.video_path_db,
                        self.video_segments,
                        self.video_segment_feature_vdb,
                        self.chunk_entity_relation_graph,
                        self.vlm_model,
                        param,
                        self._global_config(),
                    )


            else:
                raise ValueError(f"Unknown mode {self.level}")
        except Exception:
            try:
                await self._query_done()
            except Exception:
                logger.exception("Failed to persist the LLM cache after query error")
            raise
        else:
            await self._query_done()
            return response

    async def ainsert(self, new_video_segment):
        graph_snapshot = None
        new_text_chunk_ids = []
        try:
            # ---------- chunking
            inserting_chunks = get_chunks(
                new_videos=new_video_segment,
                chunk_type = 'content',
                chunk_func=self.chunk_func,
                tiktoken_model_name=self.tiktoken_model_name,
                max_token_size=self.chunk_token_size,
            )

            caption_chunks = get_chunks(new_videos=new_video_segment,
                chunk_type = 'captions',
                chunk_func=self.vector_func,
                tiktoken_model_name=self.tiktoken_model_name,
                max_token_size=self.chunk_token_size,)

            ocrs_chunks = get_chunks(new_videos=new_video_segment,
                                        chunk_type='ocrs',
                                        chunk_func=self.vector_func,
                                        tiktoken_model_name=self.tiktoken_model_name,
                                        max_token_size=self.chunk_token_size, )
            transcript_chunks = get_chunks(new_videos=new_video_segment,
                                     chunk_type='transcript',
                                     chunk_func=self.vector_func,
                                     tiktoken_model_name=self.tiktoken_model_name,
                                     max_token_size=self.chunk_token_size, )

            async def only_new_chunks(label, chunks):
                add_chunk_keys = await self.text_chunks.filter_keys(list(chunks))
                new_chunks = {
                    key: value
                    for key, value in chunks.items()
                    if key in add_chunk_keys
                }
                if new_chunks:
                    logger.info(f"[New {label}] inserting {len(new_chunks)} chunks")
                else:
                    logger.info(f"No new {label} chunks to insert")
                return new_chunks

            (
                inserting_chunks,
                caption_chunks,
                ocrs_chunks,
                transcript_chunks,
            ) = await asyncio.gather(
                only_new_chunks("content", inserting_chunks),
                only_new_chunks("caption", caption_chunks),
                only_new_chunks("OCR", ocrs_chunks),
                only_new_chunks("ASR", transcript_chunks),
            )

            vdb_upserts = []
            if inserting_chunks:
                vdb_upserts.append(self.chunks_vdb.upsert(inserting_chunks))
            if caption_chunks:
                vdb_upserts.append(self.caption_vdb.upsert(caption_chunks))
            if transcript_chunks:
                vdb_upserts.append(self.ASR_vdb.upsert(transcript_chunks))
            if ocrs_chunks:
                vdb_upserts.append(self.OCR_vdb.upsert(ocrs_chunks))
            if vdb_upserts:
                await asyncio.gather(*vdb_upserts)

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            if self.level == 'Level C' and inserting_chunks:
                graph_snapshot = self.chunk_entity_relation_graph.snapshot()
                extraction_result = await self.entity_extraction_func(
                    inserting_chunks,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entity_vdb=self.entities_vdb,
                    relationship_vdb=self.relationships_vdb,
                    global_config=self._global_config(),
                )
                if extraction_result is None:
                    logger.warning("No new entities found")
                else:
                    maybe_new_kg, _, _ = extraction_result
                    self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- commit upsertings and indexing
            all_new_chunks = {
                **inserting_chunks,
                **caption_chunks,
                **transcript_chunks,
                **ocrs_chunks,
            }
            new_text_chunk_ids = list(all_new_chunks)
            await self._insert_done(all_new_chunks)
        except Exception:
            if graph_snapshot is not None:
                self.chunk_entity_relation_graph.restore(graph_snapshot)
            if new_text_chunk_ids:
                await self.text_chunks.delete(new_text_chunk_ids)
            # LLM responses are safe to preserve, but derived indexes must not
            # be persisted until the complete insertion succeeds.
            try:
                await self._commit_storages(self.llm_response_cache)
            except Exception:
                logger.exception("Failed to persist the LLM cache after insertion error")
            raise

    async def _save_level_a_media(self):
        await self._commit_storages(self.video_dict, self.image_dict)

    async def _insert_done(self, new_text_chunks):
        if self.level=='Level A':
            await self._commit_storages(self.video_dict, self.image_dict)
            return
        elif self.level=='Level B':
            storages = [
                self.llm_response_cache,
                self.chunks_vdb,
                self.ASR_vdb,
                self.OCR_vdb,
                self.caption_vdb,
                self.video_segment_feature_vdb,
            ]
        elif self.level == 'Level C':
            storages = [
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.ASR_vdb,
                self.OCR_vdb,
                self.caption_vdb,
                self.chunk_entity_relation_graph,
                self.video_segment_feature_vdb,
            ]
        else:
            raise ValueError(f"Unknown query level: {self.level}")

        # Derived indexes are written first. text_chunks is their completion
        # marker and is persisted only after every derived store succeeds.
        await self._commit_storages(*storages)
        if new_text_chunks:
            await self.text_chunks.upsert(new_text_chunks)
            await self._commit_storages(self.text_chunks)

    async def _query_done(self):
        await self._commit_storages(self.llm_response_cache)

    async def _commit_storages(self, *storages):
        callbacks = [
            cast(StorageNameSpace, storage).index_done_callback()
            for storage in storages
            if storage is not None
        ]
        results = await asyncio.gather(*callbacks, return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError(
                f"Failed to commit {len(errors)} of {len(callbacks)} storages"
            ) from errors[0]
