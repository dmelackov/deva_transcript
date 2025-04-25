import json
import pathlib
from openai import OpenAI
from config import settings

OPENAI_API = None

SYSTEM_PROMPT = \
"""
Ты ассистент, который создает конспект по сообщениям пользователя.
Не добавляй своих мыслей, используй только информацию, которая есть в исходном тексте.
Не добавляй в начало конспекта фразы, которые не относят к конспекту.
Пиши конспект в формате markdown.

В первом сообщении будут пожелания пользователя к конспекту.
Во втором сообщении будет текст пользователя.

В тексте могут встречаться заметки пользователя: <заметка>текст</заметка>
В тексте могут встречаться изображения <изображение>Название изображения : описание изображения</изображение>

Если хочешь вставить изображение в конспект пиши как в markdown: ![](Название изображения)
Название изображения указывать только в круглых скобочках - это путь к файлу.
"""

def load_openai_model():
    global OPENAI_API
    if OPENAI_API is not None:
        return
    OPENAI_API = OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key
    )

def create_summary(user_prompt: str, content_prompt: str, output_path: pathlib.Path):
    if OPENAI_API is None:
        raise Exception("Model is not loaded")

    response = OPENAI_API.chat.completions.create(
        model=settings.openai_api_model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "user",
                "content": content_prompt
            }
        ]
    )

    return response.choices[0].message.content