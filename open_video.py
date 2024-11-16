import cv2

# 打开默认摄像头（通常是 0，可以更改为 1, 2 等以选择不同的摄像头）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取帧
    ret, frame = cap.read()
    
    if not ret:
        print("无法接收帧，结束程序")
        break

    # 显示捕获的画面
    cv2.imshow('实时摄像头捕捉', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()