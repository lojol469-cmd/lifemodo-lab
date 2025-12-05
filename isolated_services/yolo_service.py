"""
🎨 EXEMPLE COMPLET : Service de traitement d'image avec YOLO
=============================================================
Démontre toutes les features du framework :
- Validation entrée/sortie
- Logging configurable
- Gestion d'erreurs
- Métriques automatiques
- Lazy loading (modèle chargé une seule fois)

Ce service détecte des objets dans une image en utilisant YOLOv8.
Aucun conflit avec diffusers ou autres libs !
"""
from base import ServiceBase
import logging
import base64
from io import BytesIO
from typing import Dict, Any


class YOLOService(ServiceBase):
    """Service de détection d'objets avec YOLOv8."""
    
    def __init__(self):
        super().__init__(
            log_level=logging.INFO,
            log_file='/tmp/yolo_service.log'
        )
        self.model = None  # Lazy loading
    
    def validate_input(self, params: Dict[str, Any]) -> None:
        """Valider les paramètres d'entrée."""
        if 'image' not in params:
            raise ValueError("Missing required parameter: image (base64)")
        
        if not isinstance(params['image'], str):
            raise ValueError("Parameter 'image' must be a base64 string")
        
        # Optionnel : vérifier confidence threshold
        if 'confidence' in params:
            conf = params['confidence']
            if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
                raise ValueError("Parameter 'confidence' must be between 0 and 1")
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """Valider le résultat avant envoi."""
        if 'detections' not in result:
            raise ValueError("Missing required field: detections")
        
        if not isinstance(result['detections'], list):
            raise ValueError("Field 'detections' must be a list")
    
    def load_model(self):
        """Charger le modèle YOLO (lazy loading)."""
        if self.model is not None:
            return
        
        self.logger.info("🔥 Loading YOLOv8 model...")
        
        try:
            from ultralytics import YOLO
            # Utiliser modèle pré-entraîné
            self.model = YOLO('yolov8n.pt')  # Nano version (rapide)
            self.logger.info("✅ Model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Détecter objets dans l'image.
        
        Args:
            params:
                - image (str): Image encodée en base64
                - confidence (float): Seuil de confiance (default: 0.5)
                - max_detections (int): Nombre max de détections (default: 10)
        
        Returns:
            {
                'detections': [
                    {
                        'class': 'person',
                        'confidence': 0.95,
                        'bbox': [x1, y1, x2, y2]
                    },
                    ...
                ],
                'num_detections': 3,
                'image_size': [width, height]
            }
        """
        # Charger modèle si besoin
        self.load_model()
        
        # Parser paramètres
        image_b64 = params['image']
        confidence = params.get('confidence', 0.5)
        max_detections = params.get('max_detections', 10)
        
        self.logger.info(f"Processing image (confidence={confidence}, max={max_detections})")
        
        # Décoder image
        try:
            from PIL import Image
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data))
            self.logger.debug(f"Image size: {image.size}")
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")
        
        # Détection
        results = self.model(image, conf=confidence, max_det=max_detections)
        
        # Parser résultats
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extraire infos
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                class_name = result.names[cls_id]
                
                detections.append({
                    'class': class_name,
                    'confidence': round(conf, 3),
                    'bbox': [round(x, 1) for x in bbox]
                })
        
        self.logger.info(f"✅ Detected {len(detections)} objects")
        
        return {
            'detections': detections,
            'num_detections': len(detections),
            'image_size': list(image.size)
        }


if __name__ == '__main__':
    YOLOService().run()
