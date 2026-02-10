from ultralytics import YOLO
import torch

def main():
    # Vérifier GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Entraînement sur : {'GPU' if device == 0 else 'CPU'}")

    # ⚠️ IMPORTANT : Charger last.pt au lieu de yolov8s.pt
    model = YOLO(r"C:\Users\MSI\Desktop\detection2\runs\detect\runs\yolov8_littering3\weights\last.pt")
    print("✅ Reprise depuis last.pt")

    # Chemin vers data.yaml
    data_path = r"C:\Users\MSI\Desktop\detection2\littering-detction.v4i.yolov8\data.yaml"

    # Reprendre l'entraînement avec resume=True
    model.train(
        data=data_path,
        epochs=150,          # Total d'epochs voulu (pas +50, mais 200 au total)
        imgsz=640,
        batch=4,
        device=device,
        patience=50,         # Augmenté pour laisser plus de temps
        optimizer="AdamW",
        lr0=0.001,           # Learning rate réduit (divisé par 10)
        project="runs/detect/runs",
        name="yolov8_littering3",
        resume=True,         # ← CLÉ : reprend depuis le checkpoint
        cache=False,
        verbose=True,
    )

    print("✅ Entraînement terminé")

    # Évaluation
    best_model = YOLO(r"C:\Users\MSI\Desktop\detection2\runs\detect\runs\yolov8_littering3\weights\best.pt")
    metrics = best_model.val(data=data_path, split="test", imgsz=640, batch=4, device=device)

    print(f"\n📊 mAP50     : {metrics.box.map50:.3f}")
    print(f"📊 mAP50-95  : {metrics.box.map:.3f}")
    print(f"📊 Précision : {metrics.box.mp:.3f}")
    print(f"📊 Rappel    : {metrics.box.mr:.3f}")

if __name__ == "__main__":
    main()