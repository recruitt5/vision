from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)

# -------------------------------------------------------------
# 1. LOAD MODEL VÀ DỮ LIỆU HIỆU CHUẨN (CALIBRATION & HOMOGRAPHY)
# -------------------------------------------------------------
model = YOLO("YOLO model/scara.pt")

# Đọc tham số khử méo ống kính camera (nếu đã lưu từ 1. Camera calib.py)
raw_camera_matrix = None
new_camera_matrix = None
calib_dist = None
use_undistort = False

if os.path.exists("DATA/camera_calibration.npz"):
    try:
        calib_data = np.load("DATA/camera_calibration.npz")
        raw_camera_matrix = calib_data["matrix"]
        calib_dist = calib_data["distortion"]
        new_camera_matrix = calib_data.get("newcameramtx", raw_camera_matrix)

        # Kiểm tra nếu k3 quá âm sẽ bỏ qua undistort để tránh đen màn hình
        k3 = calib_dist[0][-1] if len(calib_dist[0]) >= 5 else 0
        if abs(k3) < 0.5:
            use_undistort = True
            print("[INFO] Đã load tham số khử méo camera (camera_calibration.npz).")
        else:
            print(f"[CẢNH BÁO] k3 = {k3:.4f} quá lớn làm biến dạng ảnh. Tắt khử méo ống kính!")
    except Exception as e:
        print(f"[WARNING] Lỗi load camera calibration: {e}")

# Đọc ma trận Homography (nếu đã tạo file H_matrix.npy hoặc H_matrix.npz)
H_matrix = None
if os.path.exists("DATA/H_matrix.npy"):
    H_matrix = np.load("DATA/H_matrix.npy")
    print("[INFO] Đã load ma trận Homography (H_matrix.npy).")
elif os.path.exists("DATA/H_matrix.npz"):
    H_matrix = np.load("DATA/H_matrix.npz")["H"]
    print("[INFO] Đã load ma trận Homography (H_matrix.npz).")
else:
    print("[WARNING] Chưa tìm thấy file ma trận Homography (DATA/H_matrix.npy). Tọa độ Robot sẽ tạm dùng giả định!")

def pixel_to_robot(px, py, H):
    """
    Chuyển đổi từ tọa độ Pixel (px, py) sang tọa độ thực Robot SCARA (X_robot, Y_robot) dạng mm
    """
    if H is None:
        return None, None
    pt_pixel = np.array([[[px, py]]], dtype=np.float32)
    pt_robot = cv2.perspectiveTransform(pt_pixel, H)
    rx = pt_robot[0][0][0]
    ry = pt_robot[0][0][1]
    return rx, ry

# -------------------------------------------------------------
# 2. KHỞI TẠO CAMERA & VÒNG LẶP XỬ LÝ TIME-REAL
# -------------------------------------------------------------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Không mở được Camera (ID = 2)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Khử méo ảnh nếu có dữ liệu calibration chuẩn
    if use_undistort and raw_camera_matrix is not None and calib_dist is not None:
        frame = cv2.undistort(frame, raw_camera_matrix, calib_dist, None, new_camera_matrix)

    # Chạy mô hình YOLO OBB
    results = model(frame, stream=True, verbose=False)

    for r in results:
        annotated_frame = r.plot()

        if r.obb is not None and len(r.obb) > 0:
            for idx, obb_box in enumerate(r.obb):
                coordination = obb_box.xywhr[0].tolist()
                x_px, y_px, w_px, h_px, angle_rad = coordination
                angle_deg = np.degrees(angle_rad)

                # Chuyển đổi tọa độ Pixel -> Tọa độ thực Robot (mm)
                x_robot, y_robot = pixel_to_robot(x_px, y_px, H_matrix)

                # -------------------------------------------------
                # VẼ HIỂN THỊ THÔNG TIN LÊN UI FRAME (OPENCV DISPLAY)
                # -------------------------------------------------
                # 1. Vẽ tâm vật thể
                center_pt = (int(x_px), int(y_px))
                cv2.circle(annotated_frame, center_pt, 5, (0, 0, 255), -1)

                # 2. Chuẩn bị chuỗi thông tin UI
                pixel_str = f"Pixel: ({int(x_px)}, {int(y_px)})"
                angle_str = f"Angle: {angle_deg:.1f} deg"
                if x_robot is not None and y_robot is not None:
                    robot_str = f"Robot: X={x_robot:.1f}mm, Y={y_robot:.1f}mm"
                else:
                    robot_str = "Robot: (No H matrix)"

                # 3. Vẽ ô thông tin nền màu đen chữ xanh chói cạnh vật thể
                text_x = max(10, int(x_px) - 80)
                text_y = max(30, int(y_px) - 40)
                
                # Overlay thông tin trực tiếp lên góc trên màn hình UI
                cv2.putText(annotated_frame, f"--- OBJECT {idx+1} ---", (10, 30 + idx*90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(annotated_frame, pixel_str, (10, 50 + idx*90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(annotated_frame, robot_str, (10, 70 + idx*90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(annotated_frame, angle_str, (10, 90 + idx*90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

                # In ra Terminal
                print(f"\rVật {idx+1} | Pixel: ({x_px:.1f}, {y_px:.1f}) -> Robot: (X={x_robot}, Y={y_robot}) | Góc: {angle_deg:.1f}°", end="", flush=True)
        else:
            cv2.putText(annotated_frame, "Searching objects...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"\rKhông phát hiện vật thể nào...", end="", flush=True)

    cv2.imshow("SCARA Eyes - YOLO Realtime Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
