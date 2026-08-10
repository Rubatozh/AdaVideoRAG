import asyncio
import os
import re
import shutil
import tempfile
from collections import defaultdict

from moviepy.editor import VideoFileClip, concatenate_videoclips


def _split_segment_id(segment_id):
    try:
        video_name, index = segment_id.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid video segment id: {segment_id}") from exc
    return video_name, index


def _parse_segment_time(segment):
    value = segment.get("time")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = float(value[0]), float(value[1])
    elif isinstance(value, str):
        match = re.fullmatch(
            r"\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*",
            value,
        )
        if match is None:
            raise ValueError(f"Invalid segment time: {value!r}")
        start, end = float(match.group(1)), float(match.group(2))
    else:
        raise ValueError(f"Missing structured segment time: {value!r}")

    if start < 0 or end <= start:
        raise ValueError(f"Invalid segment interval: {start}-{end}")
    return start, end


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _materialize_retrieved_videos(
    retrieved_segments,
    video_path_db,
    video_segments,
    work_dir,
):
    grouped_intervals = defaultdict(list)
    for segment_id in retrieved_segments:
        video_name, index = _split_segment_id(segment_id)
        segment = video_segments.data.get(video_name, {}).get(index)
        if segment is None:
            continue
        grouped_intervals[video_name].append(_parse_segment_time(segment))

    if not grouped_intervals:
        return [], None

    cache_root = os.path.join(work_dir, "_query_cache")
    os.makedirs(cache_root, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="retrieved_", dir=cache_root)
    output_paths = []

    try:
        for output_index, video_name in enumerate(sorted(grouped_intervals)):
            source_path = video_path_db.data.get(video_name)
            if not source_path or not os.path.isfile(source_path):
                continue

            intervals = _merge_intervals(grouped_intervals[video_name])
            with VideoFileClip(source_path) as source_video:
                clips = []
                combined = None
                try:
                    for start, end in intervals:
                        bounded_start = min(max(start, 0.0), source_video.duration)
                        bounded_end = min(
                            max(end, bounded_start), source_video.duration
                        )
                        if bounded_end <= bounded_start:
                            continue
                        clips.append(source_video.subclip(bounded_start, bounded_end))

                    if not clips:
                        continue

                    combined = clips[0]
                    if len(clips) > 1:
                        combined = concatenate_videoclips(clips, method="compose")

                    output_path = os.path.join(
                        temp_dir,
                        f"retrieved_{output_index}.mp4",
                    )
                    combined.write_videofile(
                        output_path,
                        codec="libx264",
                        audio_codec="aac",
                        verbose=False,
                        logger=None,
                    )
                    output_paths.append(output_path)
                finally:
                    if combined is not None and all(
                        combined is not clip for clip in clips
                    ):
                        combined.close()
                    for clip in clips:
                        clip.close()
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    if not output_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [], None

    # Produce one deterministic compilation so every supported MLLM receives
    # all retrieved evidence, including models whose API accepts one video only.
    if len(output_paths) > 1:
        source_clips = []
        combined = None
        compile_error = None
        combined_path = os.path.join(temp_dir, "retrieved_all.mp4")
        try:
            for path in output_paths:
                source_clips.append(VideoFileClip(path))
            combined = concatenate_videoclips(source_clips, method="compose")
            combined.write_videofile(
                combined_path,
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None,
            )
        except Exception as exc:
            compile_error = exc
        finally:
            if combined is not None:
                combined.close()
            for clip in source_clips:
                clip.close()
        if compile_error is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise compile_error
        if not os.path.isfile(combined_path):
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError("Failed to compile retrieved video clips")
        for path in output_paths:
            try:
                os.remove(path)
            except OSError:
                # The whole temporary directory is removed after generation.
                pass
        output_paths = [combined_path]
    return output_paths, temp_dir


def _collect_segment_content(retrieved_segments, video_segments):
    def sort_key(segment_id):
        video_name, index = _split_segment_id(segment_id)
        return video_name, int(index)

    content = []
    for segment_id in sorted(set(retrieved_segments), key=sort_key):
        video_name, index = _split_segment_id(segment_id)
        segment = video_segments.data.get(video_name, {}).get(index)
        if segment is None:
            continue
        content.append(f"[{segment_id}]\n{segment.get('content', '')}")
    return "\n\n".join(content)


async def _run_vlm_with_retrieved_videos(
    vlm_model,
    prompt,
    retrieved_segments,
    video_path_db,
    video_segments,
    work_dir,
):
    temp_dir = None
    try:
        if not retrieved_segments:
            raise RuntimeError("No retrieved video segment is available for generation")
        video_paths, temp_dir = await asyncio.to_thread(
            _materialize_retrieved_videos,
            retrieved_segments,
            video_path_db,
            video_segments,
            work_dir,
        )

        if not video_paths:
            raise RuntimeError("No video is available for MLLM generation")

        video_input = video_paths[0] if len(video_paths) == 1 else video_paths
        return await vlm_model.process_video(video_input, prompt)
    finally:
        if temp_dir is not None:
            await asyncio.to_thread(shutil.rmtree, temp_dir, True)


async def retrieved_segment_caption_B_pipeline(
    ori_query,
    vlm_model,
    retrieved_segments,
    video_path_db,
    video_segments,
    work_dir,
):
    segment_content = _collect_segment_content(retrieved_segments, video_segments)
    query = (
        "The caption, ASR, and OCR evidence from the relevant video segments is:\n"
        f"{segment_content}\n\nQuestion: {ori_query}"
    )
    return await _run_vlm_with_retrieved_videos(
        vlm_model,
        query,
        retrieved_segments,
        video_path_db,
        video_segments,
        work_dir,
    )


async def retrieved_segment_caption_C_pipeline(
    ori_query,
    vlm_model,
    retrieved_chunk_context,
    retrieved_segments,
    video_path_db,
    video_segments,
    entities_context,
    relations_context,
    work_dir,
):
    segment_content = _collect_segment_content(retrieved_segments, video_segments)
    query = f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources: Caption, ASR, and OCR from retrieved video segments-----
```txt
{segment_content}
```
-----Sources: Retrieved text chunks-----
```txt
{retrieved_chunk_context}
```
-----Query-----
Use the retrieved video clips and grounded text evidence to answer the question.
{ori_query}
""".strip()

    return await _run_vlm_with_retrieved_videos(
        vlm_model,
        query,
        retrieved_segments,
        video_path_db,
        video_segments,
        work_dir,
    )
