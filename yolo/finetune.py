from ultralytics import YOLO
import os

# Paths and training settings
DATA = "data.yaml"
PRETRAINED = "yolo26x.pt"
PROJECT = "runs"
NAME = "finetune_yolo26"
EPOCHS = 1  # short run for quick fine-tune; increase for real training
IMGSZ = 640
BATCH = 4
CONF = 0.1
CLASS_FILTER = [32]
VIDEO_SOURCE = "../source/stug.mp4"


def main():
    # Load model
    print("Loading model:", PRETRAINED)
    model = YOLO(PRETRAINED)

    # Train (fine-tune)
    print(f"Starting training: epochs={EPOCHS}, imgsz={IMGSZ}, batch={BATCH}")
    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name=NAME,
        project=PROJECT,
        save_period=1,
        val=False,  # skip validation to speed up and ensure weights are saved
    )

    # After training, try to find best weights
    weights_dir = os.path.join(PROJECT, NAME, "weights")
    best_pt = os.path.join(weights_dir, "best.pt")
    if not os.path.exists(best_pt):
        # fallback to last.pt
        last_pt = os.path.join(weights_dir, "last.pt")
        if os.path.exists(last_pt):
            best_pt = last_pt

    if os.path.exists(best_pt):
        print("Using weights:", best_pt)
        model = YOLO(best_pt)
    else:
        print("No trained weights found, continuing with pretrained model.")

    # Run inference on the provided video
    print("Running inference on video:", VIDEO_SOURCE)
    try:
        results = model.predict(
            source=VIDEO_SOURCE,
            save=True,
            device="0",
            conf=CONF,
            classes=CLASS_FILTER,
        )
    except Exception as e:
        print("GPU inference failed or not available, retrying on CPU. Error:", e)
        results = model.predict(
            source=VIDEO_SOURCE,
            save=True,
            device="cpu",
            conf=CONF,
            classes=CLASS_FILTER,
        )

    print("Inference finished. Check the runs/detect/predict/ folder under the yolo project for outputs.")


if __name__ == "__main__":
    main()
