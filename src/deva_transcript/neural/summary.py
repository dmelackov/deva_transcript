import json
import pathlib
from openai import OpenAI
from config import settings
from deva_transcript.neural.utils import to_plain

OPENAI_API = None

SYSTEM_PROMPT = \
"""
Ты ассистент, который создает конспект по сообщениям пользователя.
Не добавляй своих мыслей, используй только информацию, которая есть в исходном тексте.
Пиши конспект в формате markdown.

В первом сообщение будут пожелания пользователя к конспекту.
Во втором сообщении будет текст пользователя.

В тексте могут встречаться заметки пользователя: <заметка>текст</заметка>
В тексте могут встречаться изображения <изображение>Название изображения : описание изображения</изображение>

Если хочешь вставить изображение в конспект пиши: <изображение>Название изображения</изображение>
"""

def load_openai_model():
    global OPENAI_API
    if OPENAI_API is not None:
        return
    OPENAI_API = OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key
    )

def create_summary(input_path: pathlib.Path, output_path: pathlib.Path):
    if OPENAI_API is None:
        raise Exception("Model is not loaded")
    
    with input_path.open("r", encoding="utf-8") as f:
        transcript = json.load(f)

    text = to_plain(transcript)

    response = OPENAI_API.chat.completions.create(
        model=settings.openai_api_model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )

    with output_path.open("a", encoding="utf-8") as f:
        f.write(response.choices[0].message.content) # type: ignore