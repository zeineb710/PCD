from ultralytics import YOLO
import cv2

# Chemin vers ton meilleur modèle entraîné
MODEL_PATH = r"C:\Users\MSI\Desktop\detection2\runs\detect\runs\yolov8_littering3\weights\best.pt"

# Paramètres de détection
CONFIDENCE_THRESHOLD = 0.05  # seuil de confiance (0.0 à 1.0)
IOU_THRESHOLD = 0.4         # seuil IoU pour NMS

# ============================================================
# CHARGER LE MODÈLE
# ============================================================
print("🔄 Chargement du modèle...")
model = YOLO(MODEL_PATH)
print("✅ Modèle chargé avec succès")

# ============================================================
# OUVRIR LA WEBCAM
# ============================================================
# 0 = caméra par défaut du PC
# Si tu as plusieurs caméras, essaye 1, 2, etc.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Erreur : impossible d'ouvrir la webcam")
    print("   Essaye de changer VideoCapture(0) par VideoCapture(1)")
    exit()

# Résolution de la webcam (optionnel)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🎥 Webcam ouverte — appuie sur 'q' pour quitter")

# ============================================================
# BOUCLE DE DÉTECTION EN TEMPS RÉEL
# ============================================================
while True:
    # Lire une frame depuis la webcam
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Erreur de lecture de la webcam")
        break
    
    # Faire la prédiction sur la frame
    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,          # pas de logs à chaque frame
        device=0,               # GPU
    )
    
    # Dessiner les bounding boxes sur l'image
    annotated_frame = results[0].plot()
    
    # Afficher le nombre de détections
    num_detections = len(results[0].boxes)
    cv2.putText(
        annotated_frame,
        f"Detections: {num_detections}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # Afficher FPS (optionnel)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # Afficher la frame avec les détections
    cv2.imshow("YOLOv8 - Person & Trash Detection", annotated_frame)
    
    # Appuie sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# NETTOYAGE
# ============================================================
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam fermée")