import os
import pathlib
import numpy as np
import cv2
from skimage.metrics import mean_squared_error

def get_pixel_difference(img1, img2):
    """Возвращает процент различий между двумя изображениями (с учетом ресайза)"""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = cv2.absdiff(img1, img2)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.shape[0] * diff.shape[1]
    return (non_zero_count / total_pixels) * 100

def find_slide_region(frame):
    """Находит область презентации в кадре"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    slide_region = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            slide_region = (x, y, w, h)

    return slide_region


def get_video_stat(input: pathlib.Path):
    cap = cv2.VideoCapture(str(input))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return (int(total_frames), int(frame_rate))

def extract_unique_slides(input_dir: pathlib.Path, output: pathlib.Path, total_frames: int, fps: int, threshold: float = 30):
    
    hashes = []
    unique_count = 0
    frame_count = 0
    slide_region = None

    for img_path in sorted(os.listdir(input_dir)):
        frame = cv2.imread(str(input_dir / img_path))

        frame_count = int(img_path.split("_")[1].split(".")[0])
        new_slide_region = find_slide_region(frame)
        if new_slide_region is None:
            continue

        if slide_region is None:
            slide_region = new_slide_region
        if slide_region != new_slide_region:
            x1, y1, w1, h1 = slide_region
            x2, y2, w2, h2 = new_slide_region
            area1 = w1 * h1
            area2 = w2 * h2
            if abs(area1 - area2) > (max(area1, area2) * 0.1):
                slide_region = new_slide_region
        
        x, y, w, h = slide_region

        # Вырезаем область презентации
        slide = frame[y:y+h, x:x+w]

        # Фильтрация уникальных кадров
        slide_cv_gray = cv2.cvtColor(slide, cv2.COLOR_BGR2GRAY)

        slide_cv_resize = cv2.resize(slide_cv_gray, (800, 600))
        skip_flag = False
        for last_slide_cv_gray in hashes[-5:]:
            # pixel_diff = get_pixel_difference(slide_cv_gray, last_slide_cv_gray)
            score = mean_squared_error(last_slide_cv_gray, slide_cv_resize)
            if score < threshold:
                skip_flag = True
                break
        
        if skip_flag: 
            continue
        hashes.append(slide_cv_resize)
        
        unique_count += 1
        save_path = output / f"slide_{unique_count:03d}_s{int(frame_count/fps)}.png"
        cv2.imwrite(str(save_path), slide)
        yield (frame_count / fps, total_frames / fps, save_path)
    