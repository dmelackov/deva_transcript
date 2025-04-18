import pathlib
from ffmpeg_asyncio import FFmpeg

async def extract_audio_and_convert(input_path: pathlib.Path, output_path: pathlib.Path):
    ffmpeg = (
        FFmpeg()
        .input(str(input_path))
        .output(str(output_path), vn=None, ar="16k", sample_fmt="s16", ac="1", y=None)
    )
    await ffmpeg.execute()

def to_plain(input: list[dict]):
    output = ""
    for i in input:
        text = i["text"].strip("\n").strip()
        output += " " + text
        if text[-1] in [".", "?", "!"]:
            output += "\n"
    return output