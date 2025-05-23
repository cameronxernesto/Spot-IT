from roboflow import Roboflow
import os
import csv
from collections import defaultdict

evaluation_folder = 'EVALUTION FOLDER HERE'
output_csv = 'evaluation_results.csv'

rf = Roboflow(api_key="vOQ5582UEj5o9sfljRjK")
project = rf.workspace().project("spot-it-2")
model = project.version(6).model

rows = [("ID", "Class")]

for filename in os.listdir(evaluation_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(evaluation_folder, filename)
        confidence = 80

        detected_class = "None"
        while confidence >= 10:
            result = model.predict(image_path, confidence=confidence, overlap=30).json()
            predictions = result["predictions"]

            if predictions:
                class_confidences = defaultdict(list)

                for pred in predictions:
                    class_name = pred["class"]
                    pred_conf = pred["confidence"]
                    class_confidences[class_name].append(pred_conf)

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
                    confidence -= 10
            else:
                confidence -= 10

        rows.append((filename, detected_class))

# Save to CSV
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
