from ultralytics import YOLO

# Build or load YOLOv11 model
# Choose one of the following:
# model = YOLO("yolo11s.yaml")  # build new
# model = YOLO("yolo11s.pt")    # pretrained
model = YOLO("yolo11s.yaml").load("yolo11s.pt")  # transfer learning
dataset = 'DATASET PATH HERE'
output_folder = 'OUTPUT MODEL FILE HERE'
# Train the model
results = model.train(
    data=dataset,
    project=output_folder,
    name="spotit",
    epochs=100,
    imgsz=640
)
# @software{yolo11_ultralytics,
#   author = {Glenn Jocher and Jing Qiu},
#   title = {Ultralytics YOLO11},
#   version = {11.0.0},
#   year = {2024},
#   url = {https://github.com/ultralytics/ultralytics},
#   orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
#   license = {AGPL-3.0}
# }