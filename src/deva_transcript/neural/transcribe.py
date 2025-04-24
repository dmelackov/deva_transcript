import json
import pathlib
from config import settings
from faster_whisper import WhisperModel

WHISPER_MODEL = None


def load_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is not None:
        return
    WHISPER_MODEL = WhisperModel(
        settings.whisper_model_name,
        device=settings.whisper_device,
        compute_type="float16" if settings.whisper_device == "cuda" else "int8",
        cpu_threads=settings.whisper_cpu_threads
    )


def transcribe_audio(input_path: pathlib.Path, output_path: pathlib.Path):
    if WHISPER_MODEL is None:
        raise Exception("Model is not loaded")
    segments, info = WHISPER_MODEL.transcribe(
        str(input_path),
        beam_size=10,
        condition_on_previous_text=False,
        vad_filter=True,
        language="ru"
    )
    result = []
    for segment in segments:
        result.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
        yield (segment.end, info.duration)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
