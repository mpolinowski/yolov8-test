import matplotlib.pyplot as plt
import cv2

plt.figure(figsize=(16, 38))

yolo8n_night_outdoor4 = cv2.imread('yolo8n_night_outdoor4.png')
yolo8x_night_outdoor4 = cv2.imread('yolo8x_night_outdoor4.png')

yolo8n_night_outdoor3 = cv2.imread('yolo8n_night_outdoor3.png')
yolo8x_night_outdoor3 = cv2.imread('yolo8x_night_outdoor3.png')

yolo8n_night_outdoor2 = cv2.imread('yolo8n_night_outdoor2.png')
yolo8x_night_outdoor2 = cv2.imread('yolo8x_night_outdoor2.png')

yolo8n_night_outdoor = cv2.imread('yolo8n_night_outdoor.png')
yolo8x_night_outdoor = cv2.imread('yolo8x_night_outdoor.png')

yolo8n_day_outdoor2 = cv2.imread('yolo8n_day_outdoor2.png')
yolo8x_day_outdoor2 = cv2.imread('yolo8x_day_outdoor2.png')

yolo8n_day_outdoor = cv2.imread('yolo8n_day_outdoor.png')
yolo8x_day_outdoor = cv2.imread('yolo8x_day_outdoor.png')

yolo8n_day_indoor2 = cv2.imread('yolo8n_day_indoor2.png')
yolo8x_day_indoor2 = cv2.imread('yolo8x_day_indoor2.png')

yolo8n_day_indoor = cv2.imread('yolo8n_day_indoor.png')
yolo8x_day_indoor = cv2.imread('yolo8x_day_indoor.png')

ax = plt.subplot(8, 2, 1)
plt.title('yolo8n')
plt.imshow(cv2.cvtColor(yolo8n_night_outdoor4, cv2.COLOR_BGR2RGB))
plt.axis("off")

ax = plt.subplot(8, 2, 2)
plt.title('yolo8x')
plt.imshow(cv2.cvtColor(yolo8x_night_outdoor4, cv2.COLOR_BGR2RGB))
plt.axis("off")


ax = plt.subplot(8, 2, 3)
plt.title('yolo8n')
plt.imshow(cv2.cvtColor(yolo8n_night_outdoor3, cv2.COLOR_BGR2RGB))
plt.axis("off")

ax = plt.subplot(8, 2, 4)
plt.title('yolo8x')
plt.imshow(cv2.cvtColor(yolo8x_night_outdoor3, cv2.COLOR_BGR2RGB))
plt.axis("off")


ax = plt.subplot(8, 2, 5)
plt.title('yolo8n')
plt.imshow(cv2.cvtColor(yolo8n_night_outdoor2, cv2.COLOR_BGR2RGB))
plt.axis("off")

ax = plt.subplot(8, 2, 6)
plt.title('yolo8x')
plt.imshow(cv2.cvtColor(yolo8x_night_outdoor2, cv2.COLOR_BGR2RGB))
plt.axis("off")


ax = plt.subplot(8, 2, 7)
plt.title('yolo8n')
plt.imshow(cv2.cvtColor(yolo8n_night_outdoor, cv2.COLOR_BGR2RGB))
plt.axis("off")

ax = plt.subplot(8, 2, 8)
plt.title('yolo8x')
plt.imshow(cv2.cvtColor(yolo8x_night_outdoor, cv2.COLOR_BGR2RGB))
plt.axis("off")


ax = plt.subplot(8, 2, 7)
plt.title('yolo8n')
plt.imshow(cv2.cvtColor(yolo8n_day_outdoor2, cv2.COLOR_BGR2RGB))
plt.axis("off")

ax = plt.subplot(8, 2, 8)
plt.title('yolo8x')
plt.imshow(cv2.cvtColor(yolo8x_day_outdoor2, cv2.COLOR_BGR2RGB))
plt.axis("off")


ax = plt.subplot(8, 2, 7)
plt.title('yolo8n')
plt.imshow(cv2.cvtColor(yolo8n_day_outdoor, cv2.COLOR_BGR2RGB))
plt.axis("off")

ax = plt.subplot(8, 2, 8)
plt.title('yolo8x')
plt.imshow(cv2.cvtColor(yolo8x_day_outdoor, cv2.COLOR_BGR2RGB))
plt.axis("off")


ax = plt.subplot(8, 2, 9)
plt.title('yolo8n')
plt.imshow(cv2.cvtColor(yolo8n_day_indoor2, cv2.COLOR_BGR2RGB))
plt.axis("off")

ax = plt.subplot(8, 2, 10)
plt.title('yolo8x')
plt.imshow(cv2.cvtColor(yolo8x_day_indoor2, cv2.COLOR_BGR2RGB))
plt.axis("off")


ax = plt.subplot(8, 2, 11)
plt.title('yolo8n')
plt.imshow(cv2.cvtColor(yolo8n_day_indoor, cv2.COLOR_BGR2RGB))
plt.axis("off")

ax = plt.subplot(8, 2, 12)
plt.title('yolo8x')
plt.imshow(cv2.cvtColor(yolo8x_day_indoor, cv2.COLOR_BGR2RGB))
plt.axis("off")





plt.savefig("./Object_Detection_Yolov8_01.webp", bbox_inches='tight')