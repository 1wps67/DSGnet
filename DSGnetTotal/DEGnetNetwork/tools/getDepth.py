import cv2
import runDepth

img_path = r'/home/wh/lwp/hrnet-class/HRNet-Image-Classification-master/tools/testDepth.jpg'
img = cv2.imread(img_path)
if img.ndim == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

depthMap = runDepth.run(img,'/home/wh/lwp/hrnet-class/MiDaS/model-f6b98070.pt')
print(depthMap.shape)
cv2.imshow('depth_map',depthMap)
cv2.waitKey(0)