import os
import pathlib
import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from ffmpeg_asyncio import FFmpeg


YOLO_MODEL: YOLO = None  # type: ignore
CLIP_IMAGE_PROCESSOR: CLIPProcessor = None  # type: ignore
CLIP_MODEL: CLIPModel = None  # type: ignore


def load_models():
    global YOLO_MODEL
    global CLIP_IMAGE_PROCESSOR
    global CLIP_MODEL
    YOLO_MODEL = YOLO("finetuned_yolo_model/weights/best.pt")
    CLIP_IMAGE_PROCESSOR = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32") # type: ignore
    CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


def crop_frames(input_path: pathlib.Path, output_path: pathlib.Path):
    frames = list(map(lambda x: str(input_path / x), os.listdir(input_path)))
    n = 10
    chunks = [frames[i:i + n] for i in range(0, len(frames), n)]
    files_completed = 0
    for i in chunks:
        results = YOLO_MODEL.predict(i, verbose=False, stream=True)
        for res in results:
            files_completed += 1
            if res.boxes is None or len(res.boxes) == 0:
                continue
            path = pathlib.Path(res.path)
            save_one_box(
                res.boxes[0].xyxy,
                res.orig_img.copy(),
                file=output_path / pathlib.Path(path.name).with_suffix(".jpg"),
                BGR=True,
            )
            yield (files_completed, len(frames))


def get_video_stat(input: pathlib.Path):
    cap = cv2.VideoCapture(str(input))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return (int(total_frames), int(frame_rate))


def extract_unique_slides(input_dir: pathlib.Path, output_dir: pathlib.Path, total_frames: int, fps: int, threshold: float = 0.97):
    embedings = []
    unique_count = 0
    frame_count = 0

    for img_path in sorted(os.listdir(input_dir)):
        frame_count = int(img_path.split("_")[1].split(".")[0])
        image = cv2.imread(str(input_dir / img_path))
        inputs = CLIP_IMAGE_PROCESSOR(
            images=image, return_tensors='pt', padding=True)
        with torch.no_grad():
            embedding = CLIP_MODEL.get_image_features(**inputs) # type: ignore
        skip_flag = False
        for last_slide_cv_gray in embedings[-5:]:
            sim = F.cosine_similarity(embedding, last_slide_cv_gray).item()
            if sim > threshold:
                skip_flag = True
                break

        if skip_flag:
            continue
        embedings.append(embedding)

        unique_count += 1
        save_path = output_dir / \
            f"slide_{unique_count:03d}_s{int(frame_count/fps)}.png"
        cv2.imwrite(str(save_path), image)
        yield (frame_count/fps, total_frames/fps, save_path)


async def extract_key_frames(input_path: pathlib.Path, output_path: pathlib.Path):
    ffmpeg = (
        FFmpeg()
        .input(str(input_path), skip_frame="nokey")
        .output(str(output_path / 'keyframe_%05d.jpg'), vsync='0', frame_pts='true', y=None)
    )
    await ffmpeg.execute()
