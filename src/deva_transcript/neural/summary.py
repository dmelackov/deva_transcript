import pathlib
from openai import OpenAI
from config import settings

OPENAI_API = None

SYSTEM_PROMPT = \
"""
Ты ассистент, который создает полный конспект стенограммы по сообщениям пользователя.
Старайся сохранить определения, данные лектором.
Не добавляй своих мыслей, используй только данную стенограмму.
В конспекте обязательно должно быть то, на что преподаватель акцентирует внимание.
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
        text = f.read()

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