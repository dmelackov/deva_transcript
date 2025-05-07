from enum import Enum
import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from deva_p1_db.enums.task_type import TaskType



class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=f"{os.getenv('TARGET', 'dev')}.env")
    
    db_user: str = "postgres"
    db_password: str = "1234"
    db_ip: str = "postgres"
    db_port: int = 5432
    db_name: str = "vpn_db"

    rabbit_user: str = "guest"
    rabbit_password: str = "guest"
    rabbit_ip: str = "rabbitmq"
    rabbit_port: int = 5672

    minio_ip: str = "minio"
    minio_port: int = 9000
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "my-bucket"
    minio_secure: bool = False

    task_type: TaskType = TaskType.transcribe

    whisper_model_name: str = "large-v3"
    whisper_device: str = "cpu"
    whisper_cpu_threads: int = 0

    openai_api_key: str = ""
    openai_api_model_name: str = ""
    openai_base_url: str = ""
    

settings = Settings()
