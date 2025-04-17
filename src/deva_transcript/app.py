import pathlib
import time
from faststream import FastStream, Logger
from faststream.rabbit import RabbitBroker

from deva_p1_db.repositories import TaskRepository, FileRepository
from deva_p1_db.enums.task_type import TaskType
from deva_p1_db.enums.rabbit import RabbitQueuesToAi, RabbitQueuesToBack

from deva_transcript.database import Session
from deva_transcript.neural.transcribe import load_whisper_model, transcribe_audio
from deva_transcript.neural.utils import extract_audio_and_convert
from deva_transcript.s3 import S3_client
from deva_p1_db.schemas.task import TaskInput, TaskOutput
from config import settings

import tempfile

broker = RabbitBroker(
    url=f"amqp://{settings.rabbit_user}:{settings.rabbit_password}@{settings.rabbit_ip}:{settings.rabbit_port}/",
    host = settings.rabbit_ip,
    port = settings.rabbit_port

)
app = FastStream(broker)


working_queue = {
    TaskType.transcribe: RabbitQueuesToAi.transcribe_task,
    TaskType.summary: RabbitQueuesToAi.summary_task,
    TaskType.frames_extract: RabbitQueuesToAi.frames_extract_task
}

@broker.subscriber(working_queue[settings.task_type])
@broker.publisher(RabbitQueuesToBack.done_task)
async def handle(task: TaskInput,
                 session: Session,
                 s3: S3_client,
                 logger: Logger):
    task_repository = TaskRepository(session)
    file_repository = FileRepository(session)
    
    task_model = await task_repository.get_by_id(task.task_id)
    if task_model is None:
        raise Exception("Task not found")
    logger.info(f"Task {task.task_id} started")
    start_time = time.time()
    
    if settings.task_type == TaskType.transcribe:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = pathlib.Path(tmp_dir)
            input_path = temp_dir / "input.mp4"
            converted_path = temp_dir / "converted.wav"
            output_path = temp_dir / "output.json"

            s3.fget_object(settings.minio_bucket, str(task_model.origin_file_id), str(input_path))

            await extract_audio_and_convert(input_path, converted_path)
            transcribe_audio(converted_path, output_path)

            new_file = await file_repository.create(
                "transcript.json",
                "application/json",
                task_model.project,
                task=task_model
            )
            if new_file is None:
                raise Exception("File not created")

            s3.fput_object(settings.minio_bucket, str(new_file.id), str(output_path))
    logger.info(f"Task {task.task_id} end in {time.time() - start_time:.2f} seconds")
    return TaskOutput(task_id=task.task_id)

@app.after_startup
async def load_model():
    if settings.task_type == TaskType.transcribe:
        load_whisper_model()
