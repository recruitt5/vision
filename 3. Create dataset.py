# import thư viện
import cv2
import os

# mở video
cap = cv2.VideoCapture("DATA\WIN_20260916_11_58_01_Pro.mp4")

# kiểm tra xem vid mở được không
if not cap.isOpened():
    print("không mở được vid")

# lấy thông tin về fps và tổng frame đã chạy
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS = {fps}, Tổng số frame = {total_frames}")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # lưu ảnh
    if frame_count % 5 == 0:
        output_dir = ("DATA\training dataset")
        filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename, frame)
cap.release()
