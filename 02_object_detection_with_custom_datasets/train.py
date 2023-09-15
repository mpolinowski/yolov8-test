from ultralytics import YOLO
import matplotlib.pyplot as plt

dataset = 'citiscapes.yaml' # coco8.yaml

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8s.pt")  # load a pre-trained model (recommended for training)

# Use the model
model.train(data=dataset, epochs=10)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
pred = model("assets/test.jpg")  # predict on an image
plt.imsave("assets/prediction.jpg", pred)
# model.export(format='onnx', dynamic=True)  # export the model to ONNX format
# model.export(format='onnx', device=0)  # export the model to TensorRT format 
