from ultralytics import YOLO

# Load your trained model
model = YOLO("img_rec 3/runs/detect/aug_21/weights/best(125epochs).pt")

# model.predict(source=0, show=True)


model.predict(
    source="1793386000_12_A.png",
    project="results",
    name="heuristic_test_1",
    save=True
)

print(model.names)