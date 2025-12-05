"""
🚀 ISOLATED SERVICES FRAMEWORK
===============================
Pattern: Process isolation + JSON-RPC over stdio
Use case: Éviter conflits de dépendances entre packages Python

Architecture:
- Chaque service = subprocess Python indépendant
- Communication: JSON via stdin/stdout
- Isolation: Aucun import partagé = 0 conflit

Inspiré de:
- Language Server Protocol (LSP)
- JSON-RPC 2.0
- Unix philosophy: "Do one thing well"

Usage:
    class MyService(ServiceBase):
        def process(self, params):
            return {'result': 'OK'}
    
    if __name__ == '__main__':
        MyService().run()
"""
import json
import sys
import logging
import traceback
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime


class ServiceBase:
    """
    Classe de base pour un service isolé avec fonctionnalités avancées.
    
    Features:
    - ✅ Logging configurable
    - ✅ Validation des entrées/sorties
    - ✅ Gestion d'erreurs robuste
    - ✅ Métriques de performance
    - ✅ Support streaming (futures versions)
    """
    
    def __init__(self, log_level: int = logging.INFO, log_file: Optional[str] = None):
        """
        Initialize service with logging configuration.
        
        Args:
            log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR)
            log_file: Fichier de log optionnel (par défaut: stderr)
        """
        self.setup_logging(log_level, log_file)
        self.start_time = None
        self.metrics = {}
        
    def setup_logging(self, level: int, log_file: Optional[str]):
        """Configure logging pour le service."""
        handlers = []
        
        # Log vers fichier si spécifié
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        
        # Toujours log vers stderr (pas stdout pour ne pas polluer JSON)
        handlers.append(logging.StreamHandler(sys.stderr))
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_input(self, params: Dict[str, Any]) -> None:
        """
        Valider les paramètres d'entrée (à override).
        
        Raise ValueError si invalide.
        """
        pass
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """
        Valider le résultat avant envoi (à override).
        
        Raise ValueError si invalide.
        """
        pass
    
    def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traiter les paramètres et retourner le résultat.
        
        Args:
            params: Dictionnaire des paramètres d'entrée
            
        Returns:
            Dictionnaire du résultat
        """
        raise NotImplementedError("Vous devez implémenter la méthode process()")
    
    def run(self):
        """
        Point d'entrée du service - lit stdin, traite, écrit stdout.
        
        Protocol:
        - Input: JSON via stdin
        - Output: JSON via stdout
        - Errors: JSON via stdout + exit code 1
        - Logs: stderr uniquement
        """
        self.start_time = datetime.now()
        
        try:
            # Lire les paramètres depuis stdin
            self.logger.debug("Reading input from stdin...")
            params_json = sys.stdin.read()
            
            if not params_json.strip():
                raise ValueError("Empty input received")
            
            params = json.loads(params_json)
            self.logger.info(f"Received params: {list(params.keys())}")
            
            # Valider entrée
            self.validate_input(params)
            
            # Traiter
            self.logger.info("Processing request...")
            result = self.process(params)
            
            # Valider sortie
            self.validate_output(result)
            
            # Ajouter métriques
            duration = (datetime.now() - self.start_time).total_seconds()
            if 'success' not in result:
                result['success'] = True
            result['_meta'] = {
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            # Écrire le résultat
            self.logger.info(f"Request completed in {duration:.2f}s")
            print(json.dumps(result), flush=True)
            
        except json.JSONDecodeError as e:
            self._handle_error(e, "Invalid JSON input")
            sys.exit(1)
        except ValueError as e:
            self._handle_error(e, "Validation error")
            sys.exit(1)
        except Exception as e:
            self._handle_error(e, "Unexpected error")
            sys.exit(1)
    
    def _handle_error(self, exception: Exception, context: str):
        """Gérer et formater les erreurs de façon cohérente."""
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        error_result = {
            "success": False,
            "error": str(exception),
            "error_type": type(exception).__name__,
            "context": context,
            "traceback": traceback.format_exc(),
            "_meta": {
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        self.logger.error(f"{context}: {exception}")
        self.logger.debug(traceback.format_exc())
        
        print(json.dumps(error_result), flush=True)


if __name__ == "__main__":
    # Exemple d'utilisation
    class ExampleService(ServiceBase):
        def process(self, params):
            return {"result": f"Hello {params.get('name', 'World')}!"}
    
    service = ExampleService()
    service.run()
