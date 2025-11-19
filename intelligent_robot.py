#!/usr/bin/env python3
"""
🤖 SYSTÈME ROBOTIQUE INTELLIGENT 🤖
Classe principale pour la gestion des robots spécialisés par domaine
"""

import os
import json
import torch
from typing import Dict, Any, Optional, List
from datetime import datetime
import streamlit as st

# Imports des modèles
from ultralytics import YOLO
from transformers import pipeline
import torchaudio

# Import optionnel de LeRobot
try:
    import lerobot
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

class IntelligentRobot:
    """Système robotique intelligent avec Mistral comme cerveau central"""

    def __init__(self):
        self.brain = None  # Mistral model
        self.models = {}  # Domain-specific models
        self.apis = {}  # Inference APIs for each domain
        self.datasets = {}  # Available datasets by type
        self.active_domains = []
        self.has_brain = False

    def load_brain(self):
        """Charge le cerveau Mistral"""
        try:
            if not self.brain:
                # Import du loader Mistral depuis app.py
                from app import load_mistral_model
                self.brain, _ = load_mistral_model()
                self.has_brain = self.brain is not None
            return self.has_brain
        except Exception as e:
            print(f"Erreur chargement cerveau: {e}")
            return False

    def register_model(self, name, domain, model_path, api_config):
        """Enregistre un modèle spécialisé pour un domaine"""
        self.models[name] = {
            "domain": domain,
            "path": model_path,
            "api": api_config,
            "loaded": False,
            "model": None
        }
        if domain not in self.active_domains:
            self.active_domains.append(domain)

    def register_dataset(self, dataset_type, dataset_path, description):
        """Enregistre un dataset pour utilisation par les robots"""
        self.datasets[dataset_type] = {
            "path": dataset_path,
            "description": description,
            "loaded": False,
            "data": None
        }

    def load_model(self, model_name):
        """Charge un modèle spécifique"""
        if model_name not in self.models:
            return False

        model_info = self.models[model_name]
        try:
            if model_info["domain"] == "vision":
                if os.path.exists(model_info["path"]):
                    model_info["model"] = YOLO(model_info["path"])
                else:
                    # Fallback to default YOLO
                    model_info["model"] = YOLO("yolov8n.pt")
            elif model_info["domain"] == "language":
                if os.path.exists(model_info["path"]):
                    model_info["model"] = pipeline("text-classification", model=model_info["path"])
                else:
                    # Fallback pipeline
                    model_info["model"] = pipeline("text-classification")
            elif model_info["domain"] == "audio":
                if os.path.exists(model_info["path"]):
                    # Load PyTorch audio model
                    import torch.nn as nn
                    model_info["model"] = torch.load(model_info["path"])
                else:
                    # Mock audio model
                    class MockAudioModel(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.fc = nn.Linear(16000, 2)
                        def forward(self, x):
                            return self.fc(x.mean(dim=0).unsqueeze(0))
                    model_info["model"] = MockAudioModel()
            elif model_info["domain"] == "robotics":
                if LEROBOT_AVAILABLE:
                    # Import from app.py
                    from app import load_lerobot_model
                    model_info["model"] = load_lerobot_model(model_name)
                else:
                    # Mock robotics model
                    class MockRoboticsModel:
                        def select_action(self, obs):
                            return torch.randn(14)
                    model_info["model"] = MockRoboticsModel()

            model_info["loaded"] = True
            return True
        except Exception as e:
            print(f"Erreur chargement modèle {model_name}: {e}")
            return False

    def create_inference_api(self, model_name):
        """Crée une API d'inférence pour un modèle"""
        if model_name not in self.models:
            return None

        model_info = self.models[model_name]

        def api_function(input_data, **kwargs):
            """API générique pour l'inférence"""
            if not model_info["loaded"]:
                if not self.load_model(model_name):
                    return {"error": f"Impossible de charger le modèle {model_name}"}

            try:
                if model_info["domain"] == "vision":
                    results = model_info["model"](input_data, **kwargs)
                    return {"detections": results[0].boxes.data.tolist() if results else []}

                elif model_info["domain"] == "language":
                    results = model_info["model"](input_data, **kwargs)
                    return {"classification": results}

                elif model_info["domain"] == "audio":
                    # Audio inference
                    import torch
                    waveform, _ = torchaudio.load(input_data)
                    with torch.no_grad():
                        output = model_info["model"](waveform.mean(dim=0)[:16000].unsqueeze(0))
                        prediction = torch.argmax(output, dim=1).item()
                    return {"prediction": prediction}

                elif model_info["domain"] == "robotics":
                    # Robotics inference
                    from app import lerobot_test_vision_model
                    results = lerobot_test_vision_model(
                        "yolov8n.pt",  # Default vision model
                        model_info["model"],
                        input_data
                    )
                    return results

                else:
                    return {"error": f"Domaine non supporté: {model_info['domain']}"}

            except Exception as e:
                return {"error": str(e)}

        self.apis[model_name] = api_function
        return api_function

    def execute_task(self, task_description, input_data=None):
        """Exécute automatiquement une tâche en chainant les modèles selon la décision du cerveau"""
        try:
            # Analyser la tâche avec le cerveau
            decision = self.think_and_decide(task_description)

            if "error" in decision:
                return {"error": f"Erreur d'analyse: {decision['error']}"}

            results = {
                "task": task_description,
                "decision": decision,
                "executions": [],
                "final_result": None
            }

            # Préparer les données d'entrée
            current_data = input_data or {}

            # Exécuter les modèles dans l'ordre recommandé
            for model_name in decision.get("available_models", []):
                if model_name in self.models and self.models[model_name]["loaded"]:
                    model_info = self.models[model_name]

                    # Adapter les données selon le domaine
                    adapted_input = self._adapt_input_for_domain(model_info["domain"], current_data)

                    if adapted_input:
                        # Exécuter le modèle
                        execution_result = {
                            "model": model_name,
                            "domain": model_info["domain"],
                            "input": adapted_input,
                            "timestamp": datetime.now().isoformat()
                        }

                        try:
                            if model_info["domain"] == "vision":
                                # Pour vision, utiliser l'API si disponible
                                if model_name in self.apis:
                                    result = self.apis[model_name](adapted_input)
                                else:
                                    result = {"error": "API vision non disponible"}

                            elif model_info["domain"] == "language":
                                if model_name in self.apis:
                                    result = self.apis[model_name](adapted_input)
                                else:
                                    result = {"error": "API language non disponible"}

                            elif model_info["domain"] == "audio":
                                if model_name in self.apis:
                                    result = self.apis[model_name](adapted_input)
                                else:
                                    result = {"error": "API audio non disponible"}

                            elif model_info["domain"] == "robotics":
                                if model_name in self.apis:
                                    result = self.apis[model_name](adapted_input)
                                else:
                                    result = {"error": "API robotics non disponible"}

                            execution_result["result"] = result
                            execution_result["success"] = "error" not in result

                            # Utiliser le résultat comme input pour le prochain modèle
                            if execution_result["success"]:
                                current_data = self._extract_output_for_next(result, model_info["domain"])

                        except Exception as e:
                            execution_result["result"] = {"error": str(e)}
                            execution_result["success"] = False

                        results["executions"].append(execution_result)

            # Résultat final basé sur la dernière exécution réussie
            successful_executions = [e for e in results["executions"] if e["success"]]
            if successful_executions:
                results["final_result"] = successful_executions[-1]["result"]
                results["status"] = "success"
            else:
                results["final_result"] = {"error": "Aucune exécution réussie"}
                results["status"] = "failed"

            return results

        except Exception as e:
            return {"error": f"Erreur exécution tâche: {str(e)}"}

    def _adapt_input_for_domain(self, domain, input_data):
        """Adapte les données d'entrée selon le domaine"""
        try:
            if domain == "vision" and "image" in input_data:
                return input_data["image"]
            elif domain == "language" and "text" in input_data:
                return input_data["text"]
            elif domain == "audio" and "audio" in input_data:
                return input_data["audio"]
            elif domain == "robotics" and "image" in input_data:
                return input_data["image"]
            else:
                return None
        except Exception as e:
            print(f"Erreur adaptation input: {e}")
            return None

    def _extract_output_for_next(self, result, domain):
        """Extrait la sortie d'un modèle pour l'utiliser comme input du suivant"""
        try:
            if domain == "vision" and "detections" in result:
                # Convertir détections en description textuelle
                detections = result["detections"]
                description = f"Détecté {len(detections)} objets dans l'image"
                return {"text": description, "detections": detections}
            elif domain == "language" and "classification" in result:
                return {"text": str(result["classification"])}
            elif domain == "audio" and "prediction" in result:
                return {"text": f"Classification audio: {result['prediction']}"}
            elif domain == "robotics" and "lerobot_action" in result:
                return {"text": f"Action robotique exécutée: {str(result['lerobot_action'])[:100]}"}
            else:
                return result
        except Exception as e:
            print(f"Erreur extraction output: {e}")
            return result

def initialize_robot_system():
    """Initialise le système robotique avec tous les modèles disponibles"""
    robot = IntelligentRobot()

    # Chemins des modèles
    BASE_DIR = "lifemodo_data"
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LLM_DIR = os.path.join(BASE_DIR, "llms")
    ROBOTICS_DIR = os.path.join(BASE_DIR, "robotics")

    # Enregistrer les datasets disponibles
    dataset_path = os.path.join(BASE_DIR, "dataset.json")
    if os.path.exists(dataset_path):
        robot.register_dataset(
            "multimodal",
            dataset_path,
            "Dataset multimodal complet (vision, texte, audio)"
        )

    # Enregistrer les modèles par domaine
    domains_and_models = {
        "vision": [
            ("vision_yolo_trained", os.path.join(MODEL_DIR, "vision_model/weights/best.pt")),
            ("vision_yolo_default", "yolov8n.pt")
        ],
        "language": [
            ("language_transformers", os.path.join(MODEL_DIR, "language_model")),
            ("language_mistral", os.path.join(LLM_DIR, "mistral-7b"))
        ],
        "audio": [
            ("audio_pytorch", os.path.join(MODEL_DIR, "audio_model.pt"))
        ],
        "robotics": [
            ("robotics_aloha_cube", "lerobot/act_aloha_sim_transfer_cube_human"),
            ("robotics_aloha_insertion", "lerobot/act_aloha_sim_insertion_human")
        ]
    }

    # API configurations pour chaque domaine
    api_configs = {
        "vision": {"endpoint": "/api/vision/infer", "method": "POST", "input_type": "image"},
        "language": {"endpoint": "/api/language/infer", "method": "POST", "input_type": "text"},
        "audio": {"endpoint": "/api/audio/infer", "method": "POST", "input_type": "audio"},
        "robotics": {"endpoint": "/api/robotics/infer", "method": "POST", "input_type": "image"}
    }

    # Enregistrer tous les modèles
    for domain, models in domains_and_models.items():
        for model_name, model_path in models:
            if os.path.exists(model_path) or domain == "robotics" or model_path.endswith("yolov8n.pt"):
                robot.register_model(
                    model_name,
                    domain,
                    model_path,
                    api_configs[domain]
                )

    # Charger le cerveau Mistral
    robot.load_brain()

    return robot