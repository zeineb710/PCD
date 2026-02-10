from ultralytics import YOLO
import torch
import os
from pathlib import Path

def evaluate_model():
    """
    Script d'évaluation détaillée pour un modèle YOLO déjà entraîné
    """
    
    # ====== CONFIGURATION ======
    model_path = r"C:\Users\MSI\Desktop\detection2\runs\detect\runs\yolov8_littering3\weights\best.pt"  # Chemin vers votre modèle
    data_yaml = r"C:\Users\MSI\Desktop\detection2\littering-detction.v4i.yolov8\data.yaml"
    output_dir = "evaluation_results"  # Dossier pour sauvegarder les résultats
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Vérifier que le modèle existe
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable : {model_path}")
        print("💡 Vérifiez le chemin ou entraînez d'abord votre modèle")
        return
    
    print("="*60)
    print("🔍 ÉVALUATION DU MODÈLE YOLO")
    print("="*60)
    print(f"📂 Modèle       : {model_path}")
    print(f"📄 Dataset      : {data_yaml}")
    print(f"💾 Résultats    : {output_dir}/")
    print("="*60)
    
    # ====== CHARGER LE MODÈLE ======
    print("\n⏳ Chargement du modèle...")
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Modèle chargé sur : {device}")
    
    # ====== VALIDATION ======
    print("\n🔄 Lancement de la validation...")
    val_results = model.val(
        data=data_yaml,
        imgsz=640,
        batch=4,
        save_json=True,      # Sauvegarder les résultats en JSON
        save_hybrid=False,
        plots=True,          # Générer les graphiques
        project=output_dir,
        name="validation"
    )
    
    # ====== MÉTRIQUES GLOBALES ======
    print("\n" + "="*60)
    print("📊 MÉTRIQUES GLOBALES")
    print("="*60)
    print(f"mAP50-95 (global)  : {val_results.box.map:.4f}   (0.0 - 1.0)")
    print(f"mAP50 (global)     : {val_results.box.map50:.4f}   (0.0 - 1.0)")
    print(f"mAP75 (global)     : {val_results.box.map75:.4f}   (0.0 - 1.0)")
    print(f"Précision moyenne  : {val_results.box.mp:.4f}   (0.0 - 1.0)")
    print(f"Recall moyen       : {val_results.box.mr:.4f}   (0.0 - 1.0)")
    
    # ====== MÉTRIQUES PAR CLASSE ======
    print("\n" + "="*60)
    print("📋 MÉTRIQUES DÉTAILLÉES PAR CLASSE")
    print("="*60)
    
    class_names = model.names
    maps_per_class = val_results.box.maps
    
    print(f"\n{'Classe':<15} {'mAP50-95':<12} {'Performance'}")
    print("-" * 60)
    
    for class_id, class_name in class_names.items():
        map_value = maps_per_class[class_id]
        
        # Évaluation qualitative
        if map_value >= 0.8:
            perf = "🟢 Excellent"
        elif map_value >= 0.6:
            perf = "🟡 Bon"
        elif map_value >= 0.4:
            perf = "🟠 Moyen"
        else:
            perf = "🔴 Faible"
        
        print(f"{class_name:<15} {map_value:<12.4f} {perf}")
    
    # ====== MATRICE DE CONFUSION ======
    print("\n" + "="*60)
    print("🎯 MATRICE DE CONFUSION")
    print("="*60)
    
    if val_results.confusion_matrix is not None:
        confusion_path = os.path.join(output_dir, "validation", "confusion_matrix.png")
        val_results.confusion_matrix.plot(
            save_dir=os.path.join(output_dir, "validation"),
            names=list(class_names.values())
        )
        print(f"✅ Matrice sauvegardée : {confusion_path}")
        
        # Afficher la matrice en texte
        matrix = val_results.confusion_matrix.matrix
        print(f"\nMatrice de confusion (format texte) :")
        print(f"{'':>10}", end="")
        for name in class_names.values():
            print(f"{name[:8]:>10}", end="")
        print("  Background")
        
        for i, name in enumerate(class_names.values()):
            print(f"{name[:10]:>10}", end="")
            for j in range(len(matrix[i])):
                print(f"{int(matrix[i][j]):>10}", end="")
            print()
    else:
        print("⚠️  Matrice de confusion non disponible")
    
    # ====== ANALYSE DES RÉSULTATS ======
    print("\n" + "="*60)
    print("🔍 ANALYSE DES RÉSULTATS")
    print("="*60)
    
    global_map = val_results.box.map
    precision = val_results.box.mp
    recall = val_results.box.mr
    
    # Diagnostic général
    print("\n📌 Diagnostic général :")
    if global_map >= 0.7:
        print("   ✅ Excellente performance globale !")
    elif global_map >= 0.5:
        print("   ⚡ Performance correcte, améliorations possibles")
    else:
        print("   ⚠️  Performance faible, réentraînement recommandé")
    
    # Analyse précision vs recall
    print("\n📌 Analyse Précision vs Recall :")
    if precision > 0.8 and recall < 0.6:
        print("   🎯 Haute précision, faible recall")
        print("      → Le modèle est conservateur (peu de détections mais correctes)")
        print("      → Solution : Baisser le seuil de confiance ou ajouter plus de données")
    elif precision < 0.6 and recall > 0.8:
        print("   🎯 Faible précision, haut recall")
        print("      → Le modèle détecte beaucoup mais fait des erreurs")
        print("      → Solution : Augmenter le seuil de confiance ou améliorer les données")
    elif precision > 0.7 and recall > 0.7:
        print("   ✅ Bon équilibre précision/recall")
    else:
        print("   ⚠️  Précision et recall tous deux faibles")
        print("      → Solution : Augmenter la quantité/qualité des données")
    
    # Analyse par classe
    print("\n📌 Analyse par classe :")
    for class_id, class_name in class_names.items():
        map_value = maps_per_class[class_id]
        if map_value < 0.5:
            print(f"   ⚠️  '{class_name}' a des performances faibles ({map_value:.2f})")
            print(f"      → Ajoutez plus d'exemples de '{class_name}'")
            print(f"      → Vérifiez la qualité des annotations pour '{class_name}'")
    
    # ====== RECOMMANDATIONS ======
    print("\n" + "="*60)
    print("💡 RECOMMANDATIONS D'AMÉLIORATION")
    print("="*60)
    
    recommendations = []
    
    if global_map < 0.5:
        recommendations.append("1. 📸 Augmentez votre dataset (min 1000 images par classe)")
        recommendations.append("2. 🔍 Vérifiez la qualité de vos annotations")
        recommendations.append("3. 🔄 Appliquez plus d'augmentation de données")
    
    if global_map < 0.7:
        recommendations.append("4. ⏱️  Augmentez le nombre d'epochs (100-200)")
        recommendations.append("5. 🚀 Essayez un modèle plus grand (yolov8m ou yolov8l)")
    
    if any(maps_per_class[i] < 0.4 for i in range(len(maps_per_class))):
        recommendations.append("6. 🎯 Collectez plus d'exemples pour les classes faibles")
        recommendations.append("7. ⚖️  Équilibrez votre dataset entre les classes")
    
    if len(recommendations) == 0:
        print("   🎉 Votre modèle performe bien !")
        print("   📈 Pour aller plus loin :")
        print("      - Testez sur de nouvelles données réelles")
        print("      - Optimisez pour l'inférence (export ONNX/TensorRT)")
    else:
        for rec in recommendations:
            print(f"   {rec}")
    
    # ====== FICHIERS GÉNÉRÉS ======
    print("\n" + "="*60)
    print("📁 FICHIERS GÉNÉRÉS")
    print("="*60)
    print(f"Dossier : {output_dir}/validation/")
    print("   - confusion_matrix.png      : Matrice de confusion")
    print("   - confusion_matrix_normalized.png : Matrice normalisée")
    print("   - val_batch*_pred.jpg       : Prédictions visualisées")
    print("   - val_batch*_labels.jpg     : Labels réels")
    
    print("\n✅ Évaluation terminée !")
    print(f"📂 Consultez les résultats dans : {output_dir}/")

if __name__ == "__main__":
    evaluate_model()