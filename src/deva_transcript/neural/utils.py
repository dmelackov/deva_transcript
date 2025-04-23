import pathlib
from typing import NotRequired, TypedDict
from uuid import UUID
from ffmpeg_asyncio import FFmpeg
from deva_p1_db.models import Note, File


async def extract_audio_and_convert(input_path: pathlib.Path, output_path: pathlib.Path):
    ffmpeg = (
        FFmpeg()
        .input(str(input_path))
        .output(str(output_path), vn=None, ar="16k", sample_fmt="s16", ac="1", y=None)
    )
    await ffmpeg.execute()


async def extract_key_frames(input_path: pathlib.Path, output_path: pathlib.Path, fps=1):

    ffmpeg = (
        FFmpeg()
        .input(str(input_path), skip_frame="nokey")
        .output(str(output_path / 'keyframe_%05d.jpg'), vsync='0', frame_pts='true', y=None)
    )
    await ffmpeg.execute()

TranscriptEntry = TypedDict(
    'TranscriptEntry', {'start': float, 'end': float, 'text': str})


class PromptEntry(TypedDict):
    type: str
    text: str
    timestamp: float
    name: NotRequired[str]


def generate_prompt(transcript: list[TranscriptEntry], notes: list[Note], images: list[File]):
    prompt_list: list[PromptEntry] = []
    image_mapping: dict[str, UUID] = {}
    img_count = 1

    for i in transcript:
        prompt_list.append(
            {
                'type': 'text',
                'text': i['text'].strip("\n").strip(),
                'timestamp': i['start']
            }
        )
    for i in notes:
        prompt_list.append(
            {
                'type': 'note',
                'text': i.text.strip("\n").strip(),
                'timestamp': i.start_time_code
            }
        )
    for i in images:
        if i.metadata_timecode is None:
            continue
        if i.metadata_text is None:
            continue
        image_name = f"{img_count:d4}.png"
        prompt_list.append(
            {
                'type': 'image',
                'text': i.metadata_text.strip("\n").strip(),
                'name': image_name,
                'timestamp': i.metadata_timecode
            }
        )
        image_mapping[image_name] = i.id
        img_count += 1

    prompt_list.sort(key=lambda x: x['timestamp'])
    prompt = ""
    for i in prompt_list:
        if i['type'] == 'note':
            prompt += f"\n<заметка>{i['text']}</заметка>\n"
        elif i['type'] == 'image':
            if 'name' not in i:
                continue
            prompt += f"\n<изображение>{i['name']} : {i['text']}</изображение>\n"
        elif i['type'] == 'text':
            if prompt and prompt[-1] != "\n":
                prompt += " "
            prompt += i['text']
            if prompt[-1] in ['.', '!', '?']:
                prompt += '\n'
    return prompt, image_mapping
