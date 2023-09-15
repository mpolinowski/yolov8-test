from ultralytics import YOLO
import cv2 as cv
from glob import glob

# get videos
video_paths = glob("./input/*.mp4")
# print(video_paths)

# select a model
model_path = 'runs/detect/train4/weights/best.pt'
model = YOLO(model_path)


# read video file
video = cv.VideoCapture(video_paths[0])

if (video.isOpened() == False): 
    print("Error :: Cannot read video file")

# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('./output/test5.avi', fourcc, 20.0, size)

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

        # show video
        cv.imshow('frame', composed)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

out.release()
video.release()
    
# Closes all the frames
cv.destroyAllWindows()