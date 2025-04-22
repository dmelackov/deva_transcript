import pathlib
import time
from faststream import FastStream, Logger
from faststream.rabbit import RabbitBroker

from deva_p1_db.repositories import TaskRepository, FileRepository, ProjectRepository
from deva_p1_db.enums.task_type import TaskType
from deva_p1_db.enums.rabbit import RabbitQueuesToAi, RabbitQueuesToBack
from deva_p1_db.models import Task
from deva_p1_db.enums.file_type import FileTypes, resolve_file_type
from openai import project

from deva_transcript.database import Session
from deva_transcript.neural.summary import create_summary, load_openai_model
from deva_transcript.neural.transcribe import load_whisper_model, transcribe_audio
from deva_transcript.neural.utils import extract_audio_and_convert
from deva_transcript.s3 import S3_client
from deva_p1_db.schemas.task import TaskToAi, TaskToBack
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


async def task_transcribe(task_model: Task, session: Session, s3: S3_client, logger: Logger):
     file_repository = FileRepository(session)
     project_repository = ProjectRepository(session)
     with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = pathlib.Path(tmp_dir)

            source_file = task_model.project.origin_file
            input_type = resolve_file_type(source_file.file_type)

            input_path = temp_dir / f"input{input_type.extension}"
            converted_path = temp_dir / "converted.wav"
            output_path = temp_dir / "output.json"


            s3.fget_object(settings.minio_bucket, str(source_file.id), str(input_path))

            await extract_audio_and_convert(input_path, converted_path)
            for i in transcribe_audio(converted_path, output_path):
                await broker.publish(TaskToBack(task_id=task_model.id, done=False, status=str(i[0] / i[1])), RabbitQueuesToBack.progress_task)

            new_file = await file_repository.create(
                "transcript.json",
                FileTypes.text_json.internal,
                task_model.user,
                task_model.project,
                task=task_model
            )
            if new_file is None:
                raise Exception("File not created")
            s3.fput_object(settings.minio_bucket, str(new_file.id), str(output_path))

            await project_repository.add_transcription_file(task_model.project, new_file)

async def task_summary(task_model: Task, session: Session, s3: S3_client, logger: Logger):
    file_repository = FileRepository(session)
    project_repository = ProjectRepository(session)
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = pathlib.Path(tmp_dir) / "input.json"
        output_path = pathlib.Path(tmp_dir) / "output.md"
        transcript_file = task_model.project.transcription

        s3.fget_object(settings.minio_bucket, str(transcript_file), str(input_path))

        create_summary(input_path, output_path)

        new_file = await file_repository.create(
            "summary.txt",
            FileTypes.text_md.internal,
            task_model.user,
            task_model.project,
            task=task_model
        )
        if new_file is None:
            raise Exception("File not created")
        s3.fput_object(settings.minio_bucket, str(new_file.id), str(output_path))

        await project_repository.add_summary_file(task_model.project, new_file)

@broker.subscriber(working_queue[settings.task_type])
@broker.publisher(RabbitQueuesToBack.done_task)
async def handle(task: TaskToAi,
                 session: Session,
                 s3: S3_client,
                 logger: Logger):
    task_repository = TaskRepository(session)
    
    task_model = await task_repository.get_by_id(task.task_id)
    if task_model is None:
        raise Exception("Task not found")
    logger.info(f"Task {task.task_id} started")
    start_time = time.time()
    
    if settings.task_type == TaskType.transcribe:
       await task_transcribe(task_model, session, s3, logger)
    if settings.task_type == TaskType.summary:
        await task_summary(task_model, session, s3, logger)

    task_model.done = True
    await session.flush()

    logger.info(f"Task {task.task_id} end in {time.time() - start_time:.2f} seconds")
    return TaskToBack(task_id=task.task_id,
                      done=True,
                      status=None)

@app.after_startup
async def load_model():
    if settings.task_type == TaskType.transcribe:
        load_whisper_model()
    if settings.task_type == TaskType.summary:
        load_openai_model()
