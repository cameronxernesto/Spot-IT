from ultralytics import YOLO
import os
import csv
from collections import defaultdict

evaluation_folder = 'EVALUATION FOLDER HERE'
model_path = 'MODEL PATH HERE'
output_csv = 'evaluation_results.csv'

model = YOLO(model_path)
rows = [("ID", "Class")]

for filename in os.listdir(evaluation_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(evaluation_folder, filename)
        conf = 0.8

        detected_class = "None"
        while conf >= 0.1:
            results = model(image_path, conf=conf, iou=0.3, verbose=False)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                class_confidences = defaultdict(list)
                for box in boxes:
                    class_idx = int(box.cls[0].item())
                    class_name = model.names[class_idx]
                    confidence_score = float(box.conf[0].item())
                    class_confidences[class_name].append(confidence_score)

                duplicate_classes = {
                    cls: confs for cls, confs in class_confidences.items() if len(confs) >= 2
                }

                if duplicate_classes:
                    most_confident_class = max(
                        duplicate_classes.items(),
                        key=lambda item: sum(item[1]) / len(item[1])
                    )[0]
                    detected_class = most_confident_class
                    break
                else:
                    conf -= 0.1
            else:
                conf -= 0.1

        rows.append((filename, detected_class))

# Save to CSV
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
