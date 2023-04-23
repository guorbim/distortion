import cv2
import numpy as np

# 定义棋盘格的大小和角点数
board_w = 9
board_h = 6
board_n = board_w * board_h
board_sz = (board_w, board_h)

# 准备棋盘格的3D坐标
objp = np.zeros((board_n, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# 读取标定用的图像
img_dir = 'calib_imgs/'
img_names = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg']
obj_points = []  # 保存棋盘格的3D坐标
img_points = []  # 保存棋盘格的2D坐标
img_size = None
for name in img_names:
    img = cv2.imread(img_dir + name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_sz, None)
    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points.append(corners2)
        img_size = gray.shape[::-1]

# 标定相机并输出结果
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
print("camera matrix:\n", mtx)
print("distortion coefficients:\n", dist)

# 保存标定结果到XML文件
cv_file = cv2.FileStorage("distortion_params.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("intrinsic", mtx)
cv_file.write("distCoeffs", dist)
cv_file.release()
