import argparse
import asyncio
import logging
import multiprocessing
import os
import re


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def gpu_ids(value: str) -> str:
    if re.fullmatch(r"\d+(?:,\d+)*", value) is None:
        raise argparse.ArgumentTypeError(
            "GPU IDs must be comma-separated non-negative integers"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AdaVideoRAG demo pipeline")
    parser.add_argument("--query", required=True, help="Question about the input video")
    parser.add_argument(
        "--video-paths",
        nargs="+",
        required=True,
        help="One or more local video paths",
    )
    parser.add_argument(
        "--working-dir",
        default=os.getenv("ADAVIDEORAG_WORKING_DIR", "./adavideorag_cache"),
        help="Directory used for indexes and caches",
    )
    parser.add_argument("--task-gpu-ids", type=gpu_ids, default="0")
    parser.add_argument(
        "--caption-model-path",
        default=os.getenv("ADAVIDEORAG_CAPTION_MODEL_PATH", "./MiniCPM-V-2_6-int4"),
    )
    parser.add_argument(
        "--whisper-model-path",
        default=os.getenv(
            "ADAVIDEORAG_WHISPER_MODEL_PATH",
            "./faster-distil-whisper-large-v3",
        ),
    )
    parser.add_argument(
        "--ocr-languages",
        nargs="+",
        default=os.getenv("ADAVIDEORAG_OCR_LANGUAGES", "en").split(","),
    )
    parser.add_argument("--qwen-fps", type=positive_float, default=10.0)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.getenv("ADAVIDEORAG_LOG_LEVEL", "INFO").upper(),
    )

    parser.add_argument(
        "--llm-base-url",
        default=os.getenv("ADAVIDEORAG_LLM_BASE_URL", "http://localhost:9997/v1"),
    )
    parser.add_argument(
        "--vlm-base-url",
        default=os.getenv("ADAVIDEORAG_VLM_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument(
        "--intent-model",
        default=os.getenv("ADAVIDEORAG_INTENT_MODEL", "deepseek_32B"),
    )
    parser.add_argument(
        "--best-model",
        default=os.getenv("ADAVIDEORAG_BEST_MODEL", "deepseek_32B"),
    )
    parser.add_argument(
        "--cheap-model",
        default=os.getenv("ADAVIDEORAG_CHEAP_MODEL", "deepseek_7B"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("ADAVIDEORAG_EMBEDDING_MODEL", "bge-m3"),
    )
    parser.add_argument(
        "--vlm-model",
        default=os.getenv("ADAVIDEORAG_VLM_MODEL", "Qwen2.5_VL"),
    )
    return parser


def parse_level(response: str) -> str:
    match = re.search(r"###\s*(Level\s+[ABC])\s*###", response, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Intent classifier returned an invalid level: {response!r}")
    return match.group(1).title()


async def run_pipeline(args: argparse.Namespace) -> str:
    # Delay ML imports until main() has set CUDA_VISIBLE_DEVICES and process
    # settings. This avoids initializing CUDA on the wrong device.
    from adavideorag import AdaVideoRAG, QueryParam
    from adavideorag._llm import (
        LLMConfig,
        QwenCalculatorvLLM_API,
        configure_openai,
        gpt_4o_mini_complete,
        openai_embedding,
        query_get_level,
    )
    from adavideorag.prompt import PROMPTS

    llm_api_key = (
        os.getenv("ADAVIDEORAG_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )
    vlm_api_key = (
        os.getenv("ADAVIDEORAG_VLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )
    configure_openai(args.llm_base_url, llm_api_key)

    llm_config = LLMConfig(
        embedding_func_raw=openai_embedding,
        embedding_model_name=args.embedding_model,
        embedding_dim=1024,
        embedding_max_token_size=8192,
        embedding_batch_num=32,
        embedding_func_max_async=16,
        best_model_func_raw=gpt_4o_mini_complete,
        best_model_name=args.best_model,
        best_model_max_token_size=32768,
        best_model_max_async=16,
        cheap_model_func_raw=gpt_4o_mini_complete,
        cheap_model_name=args.cheap_model,
        cheap_model_max_token_size=32768,
        cheap_model_max_async=16,
    )

    classification = await query_get_level(
        args.intent_model,
        args.query,
        PROMPTS["adaptive_query"],
        api_base=args.llm_base_url,
        api_key_value=llm_api_key,
    )
    level = parse_level(classification)

    vlm_model = QwenCalculatorvLLM_API(
        qwen_fps=args.qwen_fps,
        api_key=vlm_api_key,
        base_url=args.vlm_base_url,
        model_name=args.vlm_model,
    )
    adavideorag = AdaVideoRAG(
        # CUDA_VISIBLE_DEVICES remaps the selected physical devices; ImageBind
        # therefore runs on the first logical device inside this process.
        cuda="cuda:0",
        llm=llm_config,
        working_dir=args.working_dir,
        caption_model_path=args.caption_model_path,
        whisper_model_path=args.whisper_model_path,
        ocr_languages=tuple(args.ocr_languages),
    )
    adavideorag.set_vlm_model(vlm_model)
    adavideorag.set_level(level)

    await adavideorag.insert_video(video_path_list=args.video_paths)
    return await adavideorag.aquery(
        query=args.query,
        param=QueryParam(mode="adavideorag"),
    )


def main() -> None:
    args = build_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.task_gpu_ids
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    multiprocessing.set_start_method("spawn", force=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print(asyncio.run(run_pipeline(args)))


if __name__ == "__main__":
    main()
