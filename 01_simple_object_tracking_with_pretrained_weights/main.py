from ultralytics import YOLO
import cv2 as cv
from glob import glob

# get videos
video_paths = glob("./input/*.mp4")
# print(video_paths)

# select a model

# |  |  |
# | -- | -- |
# | YOLOv8 | yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt | Detection |
# | YOLOv8-seg | yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt | Instance Segmentation |
# | YOLOv8-pose | yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt, yolov8l-pose.pt, yolov8x-pose.pt, yolov8x-pose-p6.pt | Pose/Keypoints |
# | YOLOv8-cls | yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt, yolov8l-cls.pt, yolov8x-cls.pt | Classification |

yolo8n = YOLO('yolov8n.pt')
yolo8s = YOLO('yolov8s.pt')
yolo8m = YOLO('yolov8m.pt')
yolo8l = YOLO('yolov8l.pt')
yolo8x = YOLO('yolov8x.pt')

model = yolo8x

# .
# ├── main.py
# ├── vids
# ├── yolov8l.pt 83.7M
# ├── yolov8m.pt 49.7M
# ├── yolov8n.pt 6.2M
# ├── yolov8s.pt 21.5M
# └── yolov8x.pt 130.5M


# read video file
video = cv.VideoCapture(video_paths[21])

if (video.isOpened() == False): 
    print("Error :: Cannot read video file")

# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('./output/yolo8x_night_outdoor4.avi', fourcc, 20.0, size)

# read frames
ret = True

while ret:
    ret, frame = video.read()

    if ret:
        # detect & track objects
        results = model.track(frame, persist=True)

        # plot results
        composed = results[0].plot()

        # save video
        out.write(composed)

        # # show video
        # cv.imshow('frame', composed)
        # if cv.waitKey(25) & 0xFF == ord('q'):
        #     break

out.release()
video.release()
    
# Closes all the frames
# cv.destroyAllWindows()