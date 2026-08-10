import logging
import math
import os

import easyocr
import numpy as np
from faster_whisper import WhisperModel
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from tqdm import tqdm


def encode_video(video, frame_times):
    frames = [video.get_frame(timestamp) for timestamp in frame_times]
    frames = np.stack(frames, axis=0)
    return [
        Image.fromarray(frame.astype("uint8")).resize((1280, 720))
        for frame in frames
    ]


def deal_video(
    video_path,
    video_segment_cache_path,
    video_name,
    segment_length,
    rough_num_frames_per_segment,
    fine_num_frames_per_segment,
    audio_output_format="mp3",
    vlm_model=None,
    tokenizer=None,
    video_output_format="mp4",
    whisper_model_path="./faster-distil-whisper-large-v3",
    ocr_languages=("en",),
    preprocessor_cache=None,
):
    if vlm_model is None or tokenizer is None:
        raise ValueError("Caption model and tokenizer must be loaded before indexing")
    segment_index = 0
    segment_index2name, segment_times_info = {}, {}
    ocrs = {}
    preprocessor_cache = (
        {} if preprocessor_cache is None else preprocessor_cache
    )
    ocr_cache_key = ("ocr", tuple(ocr_languages))
    ocr_reader = preprocessor_cache.get(ocr_cache_key)
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(list(ocr_languages))
        preprocessor_cache[ocr_cache_key] = ocr_reader

    with VideoFileClip(video_path) as video:
        total_video_length = float(video.duration)
        if not math.isfinite(total_video_length) or total_video_length <= 0:
            raise ValueError(f"Video has an invalid duration: {video_path}")

        # Use ceil only to create start offsets; keep the exact duration for
        # segment ends so a fractional final second is not silently dropped.
        start_times = list(range(0, math.ceil(total_video_length), segment_length))
        # Merge a tail shorter than five seconds into its preceding segment.
        if len(start_times) > 1 and (total_video_length - start_times[-1]) < 5:
            start_times = start_times[:-1]

        for start in tqdm(start_times, desc=f"Splitting Video {video_name}"):
            if start != start_times[-1]:
                end = min(start + segment_length, total_video_length)
            else:
                end = total_video_length

            subvideo = video.subclip(start, end)
            subvideo_length = subvideo.duration
            frame_times = np.linspace(
                0,
                subvideo_length,
                rough_num_frames_per_segment,
                endpoint=False,
            )
            frame_times += start

            fine_frame_times = np.linspace(
                0,
                subvideo_length,
                fine_num_frames_per_segment,
                endpoint=False,
            )
            fine_frame_times += start

            segment_id = str(segment_index)
            segment_index2name[segment_id] = (
                f"{video_name}-{segment_index}-{start}-{end}"
            )
            segment_times_info[segment_id] = {
                "frame_times": frame_times,
                "fine_frame_times": fine_frame_times,
                "timestamp": (start, end),
            }

            # Save each video segment for visual embedding.
            video_file = f"{segment_index2name[segment_id]}.{video_output_format}"
            final_path = os.path.join(video_segment_cache_path, video_file)
            if not os.path.exists(final_path):
                subvideo.write_videofile(
                    final_path,
                    codec="libx264",
                    verbose=False,
                    logger=None,
                )

            # Save audio independently so an interrupted run can resume even if
            # the segment video exists but its audio file does not.
            audio_file_base_name = segment_index2name[segment_id]
            audio_file = f"{audio_file_base_name}.{audio_output_format}"
            subaudio = subvideo.audio

            if subaudio is not None:
                final_path = os.path.join(video_segment_cache_path, audio_file)
                if not os.path.exists(final_path):
                    subaudio.write_audiofile(
                        final_path,
                        codec="mp3",
                        verbose=False,
                        logger=None,
                    )

            # Extract OCR evidence from the released rough-frame sample.
            clip_info = ""
            text_set = set()
            for time_point in frame_times:
                image = video.get_frame(float(time_point))
                ocr_results = ocr_reader.readtext(image)

                det_info = ""
                for result in ocr_results:
                    text = result[1]
                    confidence = result[2]
                    if confidence > 0.5 and text not in text_set:
                        det_info += f"{text}; "
                        text_set.add(text)
                if len(det_info) > 0:
                    clip_info += f"[time {float(time_point):.2f}s] {det_info}\n"
            ocrs[segment_id] = clip_info
            segment_index += 1

    # Extract ASR only when at least one segment contains audio. This avoids
    # loading Whisper for silent videos.
    transcripts = {}
    audio_files = {
        index: os.path.join(
            video_segment_cache_path,
            f"{segment_name}.{audio_output_format}",
        )
        for index, segment_name in segment_index2name.items()
    }
    if any(os.path.exists(audio_file) for audio_file in audio_files.values()):
        whisper_cache_key = ("whisper", whisper_model_path)
        whisper_model = preprocessor_cache.get(whisper_cache_key)
        if whisper_model is None:
            whisper_model = WhisperModel(whisper_model_path)
            whisper_model.logger.setLevel(logging.WARNING)
            preprocessor_cache[whisper_cache_key] = whisper_model
        for index in tqdm(
            segment_index2name,
            desc=f"Speech Recognition {video_name}",
        ):
            audio_file = audio_files[index]
            if not os.path.exists(audio_file):
                continue
            segments, _ = whisper_model.transcribe(audio_file)
            result = ""
            segment_global_start = segment_times_info[index]["timestamp"][0]
            for segment in segments:
                global_start = segment_global_start + segment.start
                global_end = segment_global_start + segment.end
                result += "[%.2fs -> %.2fs] %s\n" % (
                    global_start,
                    global_end,
                    segment.text,
                )
            transcripts[index] = result


    # Generate a caption enriched with ASR and OCR evidence.
    vlm_model.eval()
    captions = {}
    with VideoFileClip(video_path) as video:
        for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
            frame_times = segment_times_info[index]["frame_times"]
            video_frames = encode_video(video, frame_times)
            segment_transcript = transcripts.get(index, "")
            segment_ocr = ocrs[index]
            query = (
                "The transcript of the current video:\n"
                f"{segment_transcript}. The OCR of the current video:\n"
                f"{segment_ocr}.\nNow provide a description (caption) "
                "of the video in English."
            )
            msgs = [{"role": "user", "content": video_frames + [query]}]
            params = {"use_image_id": False, "max_slice_nums": 2}
            segment_caption = vlm_model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                **params,
            )
            captions[index] = (
                segment_caption.replace("\n", "").replace("<|endoftext|>", "")
            )

    # Assemble the schema consumed by chunking, retrieval, and graph extraction.
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        start, end = segment_times_info[index]["timestamp"]
        inserting_segments[index]["time"] = [float(start), float(end)]
        inserting_segments[index]["schema_version"] = 3
        temporal_header = (
            f"Video ID: {video_name}\n"
            f"Segment Index: {index}\n"
            f"Global Time Range: {float(start):.2f}s -> {float(end):.2f}s\n"
        )
        if index in transcripts:
            inserting_segments[index]["transcript"] = transcripts[index]
            inserting_segments[index]["content"] = (
                f"{temporal_header}Caption:\n{captions[index]}\n"
                f"Transcript:\n{transcripts[index]}\nOCR:\n{ocrs[index]}\n\n"
            )
        else:
            inserting_segments[index]["transcript"] = ""
            inserting_segments[index]["content"] = (
                f"{temporal_header}Caption:\n{captions[index]}\n"
                f"Transcript:\n\nOCR:\n{ocrs[index]}\n\n"
            )
        inserting_segments[index]["ocrs"] = ocrs[index]
        inserting_segments[index]["captions"] = captions[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index][
            "frame_times"
        ].tolist()

    return segment_index2name, inserting_segments
