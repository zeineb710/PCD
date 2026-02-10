from ultralytics import YOLO
import torch

def main():
    # Vérifier GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Entraînement sur : {device}")

    # Charger un modèle pré-entraîné
    model = YOLO("yolov8s.pt")

    # Chemin vers data.yaml
    data_path = r"C:\Users\MSI\Desktop\detection2\littering-detction.v4i.yolov8\data.yaml"

    # Lancer l'entraînement
    model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        batch=4,
        device=device,
        patience=30,
        optimizer="AdamW",
        lr0=0.01,
        project="runs",
        name="yolov8_littering",
        cache=True,
        verbose=True,
    )

    print("✅ Entraînement terminé")

    # Évaluation sur le set test
    best_model = YOLO(r"runs\yolov8_littering\weights\best.pt")
    metrics = best_model.val(data=data_path, split="test", imgsz=640, batch=4, device=device)

    print(f"\n📊 mAP50     : {metrics.box.map50:.3f}")
    print(f"📊 mAP50-95  : {metrics.box.map:.3f}")
    print(f"📊 Précision : {metrics.box.mp:.3f}")
    print(f"📊 Rappel    : {metrics.box.mr:.3f}")

if __name__ == "__main__":
    main()