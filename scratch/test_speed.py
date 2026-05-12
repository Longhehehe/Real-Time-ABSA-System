import time
from decord import VideoReader, cpu
import cv2
import os

video_path = r"c:\Users\Luc\OneDrive\Documents\Nhận dạng\Dataset_Output\Mưa đá, giông lốc gây thiệt hại tại nhiều địa phương _ VTV24\00002\video.mp4"

if not os.path.exists(video_path):
    print("Video not found.")
else:
    vr = VideoReader(video_path, ctx=cpu(0))
    n = len(vr)
    print(f"Total frames: {n}")
    
    t0 = time.time()
    for i in range(min(50, n)):
        frame = vr[i].asnumpy()
    print(f"Decord sequential 50 frames: {time.time()-t0:.2f}s")
    
    t0 = time.time()
    frames = vr.get_batch(range(min(50, n))).asnumpy()
    print(f"Decord batch 50 frames: {time.time()-t0:.2f}s")

    cap = cv2.VideoCapture(video_path)
    t0 = time.time()
    for i in range(min(50, n)):
        ret, frame = cap.read()
    print(f"CV2 sequential 50 frames: {time.time()-t0:.2f}s")
