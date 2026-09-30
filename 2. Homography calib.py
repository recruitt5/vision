# =====================================================================
# HỆ THỐNG HIỆU CHUẨN HOMOGRAPHY TỰ ĐỘNG BẰNG BÀN CỜ (CHESSBOARD)
# Dành cho Cánh tay Robot SCARA & Vision System
# =====================================================================

import cv2
import numpy as np
import os

# ---------------------------------------------------------------------
# 1. CẤU HÌNH THÔNG SỐ BÀN CỜ & CAMERA
# ---------------------------------------------------------------------
# Số góc trong (Inner Corners) của bàn cờ (Cùng thông số với 1. Camera calib.py)
CHECKERBOARD = (6, 8)  # (Cột góc trong, Hàng góc trong)

# Kích thước 1 ô vuông bàn cờ thực tế (Đơn vị: mm)
SQUARE_SIZE_MM = 25.0  

# Tọa độ Gốc (0, 0) mm thực tế mong muốn cho điểm góc đầu tiên của bàn cờ
ORIGIN_OFFSET_X = 0.0  # mm
ORIGIN_OFFSET_Y = 0.0  # mm

# ID Camera
CAMERA_ID = 1

# Đường dẫn file
CALIB_FILE = "DATA/camera_calibration.npz"
OUTPUT_H_FILE = "DATA/H_matrix.npy"

# Criteria dùng cho Sub-pixel corner refinement
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ---------------------------------------------------------------------
# 2. KHỞI TẠO TỌA ĐỘ THỰC THẾ (REAL-WORLD MM COORDINATES)
# ---------------------------------------------------------------------
cols, rows = CHECKERBOARD
pts_real_mm = np.zeros((cols * rows, 2), np.float32)
grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
pts_real_mm[:, 0] = grid[:, 0] * SQUARE_SIZE_MM + ORIGIN_OFFSET_X
pts_real_mm[:, 1] = grid[:, 1] * SQUARE_SIZE_MM + ORIGIN_OFFSET_Y

# ---------------------------------------------------------------------
# 3. LOAD THÔNG SỐ KHỬ MÉO CAMERA (CAMERA MATRIX & DISTORTION)
# ---------------------------------------------------------------------
raw_camera_matrix = None
new_camera_matrix = None
dist_coeffs = None
use_undistort = False # Mặc định tắt để tránh lỗi đen màn hình do file calib bị méo

if os.path.exists(CALIB_FILE):
    try:
        calib_data = np.load(CALIB_FILE)
        raw_camera_matrix = calib_data["matrix"]
        dist_coeffs = calib_data["distortion"]
        if "newcameramtx" in calib_data:
            new_camera_matrix = calib_data["newcameramtx"]
        else:
            new_camera_matrix = raw_camera_matrix

        # Kiểm tra nhanh hệ số distortion xem có bị biến dạng cực đoan (lỗi k3 quá âm) không
        k3 = dist_coeffs[0][-1] if len(dist_coeffs[0]) >= 5 else 0
        if abs(k3) > 0.5:
            print(f"[CẢNH BÁO] Hệ số méo ống kính k3 = {k3:.4f} quá lớn! Có thể làm đen hình.")
            print(" -> Mặc định đã TẮT khử méo (Undistort = OFF). Nhấn phím 'u' nếu muốn bật thử.")
            use_undistort = False
        else:
            use_undistort = True
            print(f"[INFO] Đã load thông số Ma trận Camera từ '{CALIB_FILE}'. (Undistort = ON)")
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi đọc file calib: {e}. Sẽ chạy chế độ ảnh gốc.")
else:
    print(f"[WARNING] Không tìm thấy '{CALIB_FILE}'. Sẽ chạy chế độ ảnh gốc.")

# ---------------------------------------------------------------------
# 4. MỞ CAMERA & THỰC HIỆN HIỆU CHUẨN TỰ ĐỘNG
# ---------------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"[LỖI] Không thể mở Camera ID = {CAMERA_ID}!")
    exit()

cv2.namedWindow("Automated Homography Calibration", cv2.WINDOW_AUTOSIZE)

print("=" * 65)
print(" HƯỚNG DẪN HIỆU CHUẨN HOMOGRAPHY TỰ ĐỘNG:")
print(f" 1. Đặt bàn cờ Chessboard ({cols}x{rows} góc trong, ô {SQUARE_SIZE_MM}mm) phẳng trên bàn.")
print(" 2. Đưa bàn cờ vào tầm nhìn Camera.")
print(" 3. Nhấn phím 'u' để BẬT/TẮT khử méo ống kính (Undistort).")
print(" 4. Nhấn phím 'c' hoặc 'SPACE' để TÍNH & LƯU ma trận Homography.")
print(" 5. Nhấn phím 'q' hoặc 'ESC' để thoát.")
print("=" * 65)

H_latest = None
rmse_latest = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[LỖI] Không thể đọc frame từ camera.")
        break

    # 1. Khử méo thấu kính nếu được bật và có đủ tham số
    if use_undistort and raw_camera_matrix is not None and dist_coeffs is not None:
        display_frame = cv2.undistort(frame, raw_camera_matrix, dist_coeffs, None, new_camera_matrix)
    else:
        display_frame = frame.copy()

    gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

    # 2. Tự động tìm góc bàn cờ (Chessboard Corners)
    found, corners = cv2.findChessboardCorners(
        gray, 
        CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    status_color = (0, 0, 255) # Đỏ = Chưa tìm thấy
    status_text = "Searching Chessboard..."

    if found:
        # 3. Tối ưu tọa độ góc đến độ chính xác Sub-pixel (tới 0.1 pixel)
        corners_subpixel = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        pts_pixel_arr = corners_subpixel.reshape(-1, 2)

        # 4. Tính toán Ma trận Homography H bằng RANSAC
        H, mask = cv2.findHomography(pts_pixel_arr, pts_real_mm, cv2.RANSAC, 5.0)

        if H is not None:
            H_latest = H
            
            # Tính độ sai số Reprojection Error (RMSE mm)
            pts_pix_reshaped = pts_pixel_arr.reshape(-1, 1, 2)
            pts_real_pred = cv2.perspectiveTransform(pts_pix_reshaped, H).reshape(-1, 2)
            errors_mm = np.linalg.norm(pts_real_pred - pts_real_mm, axis=1)
            rmse_latest = np.sqrt(np.mean(errors_mm ** 2))

            status_color = (0, 255, 0) # Xanh lá = Thành công
            status_text = f"Chessboard DETECTED | RMSE: {rmse_latest:.3f} mm"

            # 5. Vẽ lưới góc lên ảnh
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners_subpixel, found)

            # 6. Vẽ Gốc tọa độ (0,0) mm và hướng trục X, Y để kiểm tra thị giác
            origin_pixel = tuple(pts_pixel_arr[0].astype(int))
            pt_x = tuple(pts_pixel_arr[1].astype(int)) # Điểm kế tiếp theo chiều X
            pt_y = tuple(pts_pixel_arr[cols].astype(int)) # Điểm kế tiếp theo chiều Y

            # Trục X (Màu Đỏ)
            cv2.arrowedLine(display_frame, origin_pixel, pt_x, (0, 0, 255), 2, tipLength=0.3)
            cv2.putText(display_frame, "X(mm)", (pt_x[0] + 5, pt_x[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Trục Y (Màu Xanh Lá)
            cv2.arrowedLine(display_frame, origin_pixel, pt_y, (0, 255, 0), 2, tipLength=0.3)
            cv2.putText(display_frame, "Y(mm)", (pt_y[0], pt_y[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Đánh dấu gốc tọa độ
            cv2.circle(display_frame, origin_pixel, 8, (255, 0, 255), -1)
            cv2.putText(display_frame, f"ORIGIN (0,0)mm", (origin_pixel[0] + 10, origin_pixel[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # -----------------------------------------------------------------
    # UI OVERLAY (HIỂN THỊ TRẠNG THÁI VÀ THÔNG TIN)
    # -----------------------------------------------------------------
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 75), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

    undist_str = "ON" if use_undistort else "OFF (Press 'U' to toggle)"
    cv2.putText(display_frame, f"STATUS: {status_text}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(display_frame, f"Config: Grid {cols}x{rows} | Square: {SQUARE_SIZE_MM}mm | Undistort: {undist_str}", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if H_latest is not None:
        cv2.putText(display_frame, "[Press 'C' / 'SPACE' to SAVE Matrix]", (display_frame.shape[1] - 330, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("Automated Homography Calibration", display_frame)

    # -----------------------------------------------------------------
    # XỬ LÝ SỰ KIỆN PHÍM BẤM
    # -----------------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    # Phím 'u' để Bật/Tắt Undistort
    if key == ord('u'):
        use_undistort = not use_undistort
        print(f"[TOGGLE] Undistort mode: {'ON' if use_undistort else 'OFF'}")

    # Phím 'c' hoặc SPACE để Lưu Ma trận H
    if key == ord('c') or key == 32:
        if H_latest is not None:
            os.makedirs(os.path.dirname(OUTPUT_H_FILE), exist_ok=True)
            np.save(OUTPUT_H_FILE, H_latest)
            print("\n" + "=" * 60)
            print(f"[THÀNH CÔNG] Đã lưu ma trận Homography H vào '{OUTPUT_H_FILE}'!")
            print(f"Độ sai số hiệu chuẩn (RMSE): {rmse_latest:.4f} mm")
            print("Ma trận H (Pixel -> Real mm):\n", H_latest)
            print("=" * 60 + "\n")
            cv2.putText(display_frame, "SAVED SUCCESSFULLY!", (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow("Automated Homography Calibration", display_frame)
            cv2.waitKey(1500)
            break
        else:
            print("\n[CẢNH BÁO] Chưa phát hiện được bàn cờ! Vui lòng căn chỉnh bàn cờ trước khi bấm lưu.")

    # Phím 'q' hoặc ESC để thoát
    if key == ord('q') or key == 27:
        print("\n[INFO] Đã thoát chương trình mà không lưu H mới.")
        break

cap.release()
cv2.destroyAllWindows()