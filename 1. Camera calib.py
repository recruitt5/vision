# Import required modules
import cv2
import numpy as np
import os
import glob

# =====================================================================
# NỘI SUY THÔNG SỐ VÀ MA TRẬN CAMERA LENSE
# =====================================================================

# Define the dimensions of checkerboard
CHECKERBOARD = (6, 8)
# Criteria for stopping
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Vector for 3D points
threedpoints = []
# Vector for 2D points
twodpoints = []
# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# Capturing video
cap = cv2.VideoCapture(1)

# Vòng lặp vô hạn
while True: 
    ret, image = cap.read()
    if not ret:
        break

    # Make image gray
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get image size
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Tìm đủ số góc
    found, corners = cv2.findChessboardCorners( grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE )

   # nếu mọi góc đều tìm được, subpix coor và vẽ góc lên hình
    corners2 = None
    if found: 
        corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
        image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, found)

    # UI for saved frames
    cv2.putText(image, f"Saved frames: {len(threedpoints)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ĐƯA RA NGOÀI if found ĐỂ MÀN HÌNH LUÔN UPDATE VÀ BẮT ĐƯỢC PHÍM BẤM
    cv2.imshow('Live Calibration', image) 
    

    # Press space to store a frame, world coor and corner coor
    key = cv2.waitKey(1) & 0xFF
    if key == 32 and found and corners2 is not None:
        threedpoints.append(objectp3d)
        twodpoints.append(corners2)
        print(f"Đã lưu frame số: {len(threedpoints)}")

    # Bấm 'q' hoặc ESC (mã 27) để kết thúc thu thập
    if key == ord('q') or key == 27:
        break
cap.release()
cv2.destroyAllWindows()

# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
print(f"\nĐang tính toán hiệu chuẩn trên {len(threedpoints)} frame đã lưu...")
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints, twodpoints, grayColor.shape[::-1], None, None)

# Tính ma trận camera tối ưu (Optimal New Camera Matrix)
h, w = grayColor.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h))

# Displaying required output
print(f"Độ phân giải camera đang mở: {int(width)}x{int(height)}")
print(" Camera matrix:")
print(matrix)
print("Optimized camera matrix: ")
print(newcameramtx)
print("\n Distortion coefficient:")
print(distortion)
print("\n Rotation Vectors:")
print(r_vecs)
print("\n Translation Vectors:")
print(t_vecs)
print(" RMS error:")
print(ret)
# Lưu trữ dữ liệu
np.savez("DATA/camera_calibration.npz", matrix=matrix, distortion=distortion, newcameramtx=newcameramtx)
print("Đã lưu thông số calibration vào file 'camera_calibration.npz'!")




