import json
import pathlib
import time
from faststream import FastStream, Logger
from faststream.rabbit import RabbitBroker

from deva_p1_db.repositories import TaskRepository, FileRepository, ProjectRepository, NoteRepository
from deva_p1_db.enums.task_type import TaskType
from deva_p1_db.enums.rabbit import RabbitQueuesToAi, RabbitQueuesToBack
from deva_p1_db.models import Task
from deva_p1_db.enums.file_type import FileTypes, resolve_file_type
from openai import project

from deva_transcript.database import Session
import deva_transcript.neural.frames_extract as frames_extract
from deva_transcript.neural.summary import create_summary, load_openai_model
from deva_transcript.neural.transcribe import load_whisper_model, transcribe_audio
from deva_transcript.neural.utils import extract_audio_and_convert, extract_key_frames, generate_prompt
from deva_transcript.s3 import S3_client
from deva_p1_db.schemas.task import TaskToAi, TaskReadyToBack, TaskStatusToBack, TaskErrorToBack
from config import settings

import tempfile

broker = RabbitBroker(
    url=f"amqp://{settings.rabbit_user}:{settings.rabbit_password}@{settings.rabbit_ip}:{settings.rabbit_port}/",
    host=settings.rabbit_ip,
    port=settings.rabbit_port

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

        if source_file is None:
            raise Exception("Source file not found")
        input_type = resolve_file_type(source_file.file_type)

        input_path = temp_dir / f"input{input_type.extension}"
        converted_path = temp_dir / "converted.wav"
        output_path = temp_dir / "output.json"

        s3.fget_object(settings.minio_bucket,
                       source_file.minio_name, str(input_path))

        await extract_audio_and_convert(input_path, converted_path)
        for i in transcribe_audio(converted_path, output_path):
            await broker.publish(TaskStatusToBack(task_id=task_model.id, progress=i[0] / i[1]), RabbitQueuesToBack.progress_task)

        new_file = await file_repository.create(
            task_model.user,
            task_model.project,
            "transcript.json",
            FileTypes.text_json.internal,
            file_size=0,
            task=task_model
        )
        if new_file is None:
            raise Exception("File not created")
        s3.fput_object(settings.minio_bucket,
                       new_file.minio_name,
                       str(output_path),
                       content_type=FileTypes.text_json.mime)

        await project_repository.add_transcription_file(task_model.project, new_file)


async def task_summary(task_model: Task, session: Session, s3: S3_client, logger: Logger):
    file_repository = FileRepository(session)
    project_repository = ProjectRepository(session)
    note_repository = NoteRepository(session)

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = pathlib.Path(tmp_dir) / "input.json"
        output_path = pathlib.Path(tmp_dir) / "output.md"
        transcript_file = task_model.project.transcription
        if transcript_file is None:
            raise Exception("Source file not found")
        if task_model.project.origin_file is None:
            raise Exception("Source file not found")

        s3.fget_object(settings.minio_bucket,
                       transcript_file.minio_name, str(input_path))

        images = await file_repository.get_active_images(task_model.project)
        notes = await note_repository.get_by_file(task_model.project.origin_file)

        content_prompt, image_mapping = generate_prompt(
            json.load(open(input_path)), notes, images)
        logger.info("Content Prompt:\n" + content_prompt)
        summary = create_summary(
            task_model.prompt, content_prompt, output_path)

        if summary is None:
            raise Exception("Summary not generated")

        for k, v in image_mapping.items():
            summary = summary.replace(k, str(v))

        with output_path.open(mode="w", encoding="utf-8") as f:
            f.write(summary)
            f.flush()

        new_file = await file_repository.create(
            task_model.user,
            task_model.project,
            "summary.md",
            FileTypes.text_md.internal,
            file_size=0,
            task=task_model
        )
        if new_file is None:
            raise Exception("File not created")
        s3.fput_object(settings.minio_bucket,
                       new_file.minio_name,
                       str(output_path),
                       content_type=FileTypes.text_md.mime)

        await project_repository.add_summary_file(task_model.project, new_file)


async def frames_extract_task(task_model: Task, session: Session, s3: S3_client, logger: Logger):
    file_repository = FileRepository(session)
    project_repository = ProjectRepository(session)
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_dir = pathlib.Path(tmp_dir)

        source_file = task_model.project.origin_file
        if source_file is None:
            raise Exception("Source file not found")
        input_type = resolve_file_type(source_file.file_type)

        input_path = temp_dir / f"input{input_type.extension}"
        converted_dir = temp_dir / "frames"
        croped_dir = temp_dir / "crops"
        output_dir = temp_dir / "images"

        converted_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        croped_dir.mkdir(exist_ok=True)

        s3.fget_object(settings.minio_bucket,
                       source_file.minio_name, str(input_path))

        await extract_key_frames(input_path, converted_dir)

        images: list[tuple[float, pathlib.Path]] = []

        total_frames, fps = frames_extract.get_video_stat(input_path)
        await frames_extract.extract_key_frames(input_path, converted_dir)
        await broker.publish(
            TaskStatusToBack(task_id=task_model.id, progress=1/3), RabbitQueuesToBack.progress_task)
        last_time = time.time()
        for i in frames_extract.crop_frames(converted_dir, croped_dir):
            if time.time() - last_time > 3:
                await broker.publish(
                    TaskStatusToBack(task_id=task_model.id, progress=1/3 + i[0]/i[1]*1/3), RabbitQueuesToBack.progress_task)
                last_time = time.time()
        for i in frames_extract.extract_unique_slides(input_dir=croped_dir, output_dir=output_dir, total_frames=total_frames, fps=fps):
            if time.time() - last_time > 3:
                await broker.publish(
                    TaskStatusToBack(task_id=task_model.id, progress=2/3 + i[0]/i[1]*1/3), RabbitQueuesToBack.progress_task)
                last_time = time.time()
            images.append((i[0], i[2]))

        for i in images:
            new_file = await file_repository.create(
                task_model.user,
                task_model.project,
                i[1].name,
                FileTypes.image_png.internal,
                file_size=0,
                task=task_model,
                metadata_timecode=i[0],
                metadata_is_hide=False,
                metadata_text=""
            )
            if new_file is None:
                raise Exception("File not created")
            s3.fput_object(settings.minio_bucket,
                           new_file.minio_name,
                           str(i[1]),
                           content_type=FileTypes.image_png.mime)

        await project_repository.frames_extracted_done(task_model.project)


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
    try:
        if settings.task_type == TaskType.transcribe:
            await task_transcribe(task_model, session, s3, logger)
        if settings.task_type == TaskType.summary:
            await task_summary(task_model, session, s3, logger)
        if settings.task_type == TaskType.frames_extract:
            await frames_extract_task(task_model, session, s3, logger)
    except Exception as e:
        logger.error(f"Task {task.task_id} failed: {e}")
        await broker.publish(TaskErrorToBack(task_id=task.task_id, error=str(e)), RabbitQueuesToBack.error_task)

    task_model.done = True
    await session.flush()
    await session.commit()

    logger.info(
        f"Task {task.task_id} end in {time.time() - start_time:.2f} seconds")
    return TaskReadyToBack(task_id=task.task_id)

if settings.task_type == TaskType.summary:
    @broker.subscriber(RabbitQueuesToAi.summary_edit_task)
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

        pass  # TODO

        task_model.done = True
        await session.flush()

        logger.info(
            f"Task {task.task_id} end in {time.time() - start_time:.2f} seconds")
        return TaskReadyToBack(task_id=task.task_id)


@app.after_startup
async def load_model():
    if settings.task_type == TaskType.transcribe:
        load_whisper_model()
    if settings.task_type == TaskType.summary:
        load_openai_model()
    if settings.task_type == TaskType.frames_extract:
        frames_extract.load_models()
