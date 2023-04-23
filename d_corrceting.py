import cv2
import os

# 加载相机畸变参数
distorion_params = cv2.FileStorage("distortion_params.xml", cv2.FILE_STORAGE_READ)
camera_matrix = distorion_params.getNode("intrinsic").mat()
dist_coeffs = distorion_params.getNode("distCoeffs").mat()

# 确保输出目录存在
input = "input"
output = "output"
if not os.path.exists(output):
    os.makedirs(output)

# 遍历原始图像目录下的所有图片文件
for filename in os.listdir(input):
    if not filename.endswith(".jpg"):
        continue

    # 加载原始图像
    img = cv2.imread(os.path.join(input, filename))

    # 进行畸变矫正
    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)

    # 保存输出结果图像文件为 jpg 格式
    output_filename = os.path.join(output, filename)
    cv2.imwrite(output_filename, img_undistorted)
