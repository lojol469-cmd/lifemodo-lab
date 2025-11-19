#!/usr/bin/env python3
"""
Démonstration du Système Robotique Intelligent avec Mistral Brain
"""
import os
import sys
import json

# Ajouter le répertoire parent au path pour importer app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_robot_system():
    """Démontre les capacités du système robotique intelligent"""

    print("🤖 === DÉMONSTRATION SYSTÈME ROBOTIQUE INTELLIGENT === 🤖\n")

    # Importer les classes
    try:
        from app import IntelligentRobot, initialize_robot_system
        print("✅ Import des classes réussi")
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        return

    # Créer et initialiser le robot
    print("\n🔄 Initialisation du système robotique...")
    robot = IntelligentRobot()

    # Enregistrer des modèles de démonstration
    print("📝 Enregistrement des modèles par domaine...")

    # Modèles de vision
    robot.register_model(
        "vision_yolo_trained",
        "vision",
        "lifemodo_data/models/vision_model/weights/best.pt",
        {"endpoint": "/api/vision/infer", "method": "POST", "input_type": "image"}
    )

    robot.register_model(
        "vision_yolo_default",
        "vision",
        "yolov8n.pt",
        {"endpoint": "/api/vision/infer", "method": "POST", "input_type": "image"}
    )

    # Modèles de langage
    robot.register_model(
        "language_transformers",
        "language",
        "lifemodo_data/models/language_model",
        {"endpoint": "/api/language/infer", "method": "POST", "input_type": "text"}
    )

    # Modèles audio
    robot.register_model(
        "audio_pytorch",
        "audio",
        "lifemodo_data/models/audio_model.pt",
        {"endpoint": "/api/audio/infer", "method": "POST", "input_type": "audio"}
    )

    # Modèles robotiques
    robot.register_model(
        "robotics_aloha_cube",
        "robotics",
        "lerobot/act_aloha_sim_transfer_cube_human",
        {"endpoint": "/api/robotics/infer", "method": "POST", "input_type": "image"}
    )

    print(f"✅ {len(robot.models)} modèles enregistrés dans {len(robot.active_domains)} domaines")

    # Enregistrer des datasets
    print("\n📊 Enregistrement des datasets...")
    robot.register_dataset(
        "multimodal",
        "lifemodo_data/dataset.json",
        "Dataset multimodal complet (vision, texte, audio)"
    )

    print(f"✅ {len(robot.datasets)} datasets enregistrés")

    # Afficher l'état du système
    print("\n📋 État du système robotique:")
    print(f"   🧠 Cerveau: {'✅ Chargé' if robot.brain else '❌ Non chargé'}")
    print(f"   🤖 Modèles: {len(robot.models)}")
    print(f"   🎯 Domaines: {', '.join(robot.active_domains)}")
    print(f"   📊 Datasets: {len(robot.datasets)}")

    # Démonstration de la prise de décision intelligente
    print("\n🧠 Démonstration de l'analyse intelligente...")

    test_tasks = [
        "Analyse cette image et décris ce que tu vois",
        "Écoute cet audio et transcris le en texte",
        "Lis ce document et résume le contenu",
        "Vois cette scène et simule une action robotique pour saisir l'objet"
    ]

    for task in test_tasks:
        print(f"\n🎯 Tâche: '{task}'")

        # Simuler l'analyse (sans Mistral pour la démo)
        print("🤔 Analyse: Cette tâche nécessite...")

        # Déterminer le domaine basé sur les mots-clés
        if any(word in task.lower() for word in ["image", "vois", "regarde", "visualise"]):
            domain = "vision"
            models = [name for name, info in robot.models.items() if info["domain"] == "vision"]
        elif any(word in task.lower() for word in ["écoute", "audio", "son"]):
            domain = "audio"
            models = [name for name, info in robot.models.items() if info["domain"] == "audio"]
        elif any(word in task.lower() for word in ["lis", "texte", "document"]):
            domain = "language"
            models = [name for name, info in robot.models.items() if info["domain"] == "language"]
        elif any(word in task.lower() for word in ["robot", "action", "saisir", "manipule"]):
            domain = "robotics"
            models = [name for name, info in robot.models.items() if info["domain"] == "robotics"]
        else:
            domain = "unknown"
            models = []

        print(f"   📍 Domaine identifié: {domain}")
        print(f"   🤖 Modèles disponibles: {models}")

        if models:
            recommended_model = models[0]  # Prendre le premier disponible
            print(f"   ✅ Modèle recommandé: {recommended_model}")
            print(f"   🔗 API: {robot.models[recommended_model]['api']['endpoint']}")
        else:
            print("   ⚠️ Aucun modèle disponible pour ce domaine")

    # Démonstration des APIs
    print("\n🔌 APIs d'inférence disponibles:")
    for model_name, model_info in robot.models.items():
        api = model_info["api"]
        print(f"   • {model_name}: {api['endpoint']} ({api['input_type']})")

    # Export de configuration
    print("\n📤 Export de la configuration...")

    config = {
        "system": "Intelligent Robot System v1.0",
        "brain": "Mistral-7B",
        "models": {
            name: {
                "domain": info["domain"],
                "api_endpoint": info["api"]["endpoint"],
                "input_type": info["api"]["input_type"]
            }
            for name, info in robot.models.items()
        },
        "domains": robot.active_domains,
        "datasets": list(robot.datasets.keys())
    }

    config_file = "robot_config_demo.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ Configuration exportée vers: {config_file}")

    # Résumé final
    print("\n🎉 === RÉSUMÉ DU SYSTÈME ROBOTIQUE === 🎉")
    print("✅ Architecture modulaire par domaine")
    print("✅ Cerveau Mistral pour prise de décision")
    print("✅ APIs d'inférence spécialisées")
    print("✅ Support multi-dataset")
    print("✅ Interface utilisateur intuitive")
    print("✅ Export de configuration")
    print("\n🚀 Le système est prêt pour utilisation dans vos domaines spécifiques!")

if __name__ == "__main__":
    demo_robot_system()