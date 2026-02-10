from ultralytics import YOLO

# Charger ton modèle entraîné
model = YOLO(r"C:\Users\MSI\Desktop\detection2\runs\detect\runs\yolov8_littering3\weights\best.pt")

# Image à tester
image_path = r"C:\Users\MSI\Desktop\detection2\test3.webp"

# Prédiction
results = model.predict(
    source=image_path,
    conf=0.1,
    show=True,     # affiche l'image avec les boxes
    save=True      # sauvegarde le résultat
)

print("✅ Test image terminé")
