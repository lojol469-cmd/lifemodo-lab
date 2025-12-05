"""
Client pour appeler des services isolés via subprocess.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class IsolatedService:
    """Client pour appeler un service Python isolé."""
    
    def __init__(self, name: str, script_path: str, python_executable: str = None):
        """
        Initialiser le client de service isolé.
        
        Args:
            name: Nom du service (pour logging)
            script_path: Chemin vers le script Python du service
            python_executable: Chemin vers l'exécutable Python (défaut: sys.executable)
        """
        self.name = name
        self.script_path = Path(script_path)
        self.python_executable = python_executable or sys.executable
        
        if not self.script_path.exists():
            raise FileNotFoundError(f"Script de service introuvable: {self.script_path}")
    
    def call(self, params: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Appeler le service avec les paramètres donnés.
        
        Args:
            params: Dictionnaire des paramètres
            timeout: Timeout en secondes (None = pas de timeout)
            
        Returns:
            Dictionnaire du résultat
            
        Raises:
            RuntimeError: Si le service échoue
            TimeoutError: Si le timeout est dépassé
        """
        try:
            # Convertir les paramètres en JSON
            params_json = json.dumps(params)
            
            # Appeler le service
            result = subprocess.run(
                [self.python_executable, str(self.script_path)],
                input=params_json,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Vérifier le code de retour
            if result.returncode != 0:
                error_msg = f"Service {self.name} échoué (code {result.returncode})"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                raise RuntimeError(error_msg)
            
            # Parser la sortie
            if not result.stdout.strip():
                raise RuntimeError(f"Service {self.name} n'a retourné aucune sortie")
            
            try:
                output = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Service {self.name} sortie JSON invalide: {e}")
            
            # Vérifier les erreurs dans la sortie
            if isinstance(output, dict) and "error" in output:
                raise RuntimeError(f"Service {self.name} erreur: {output['error']}")
            
            return output
            
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Service {self.name} timeout après {timeout}s")
        except Exception as e:
            if isinstance(e, (RuntimeError, TimeoutError)):
                raise
            raise RuntimeError(f"Service {self.name} erreur inattendue: {e}")


# Fonction helper pour créer rapidement un service
def create_service(name: str, script_path: str, **kwargs) -> IsolatedService:
    """Helper pour créer un service isolé."""
    return IsolatedService(name, script_path, **kwargs)
