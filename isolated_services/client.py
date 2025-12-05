"""
🚀 CLIENT HELPER POUR ISOLATED SERVICES
========================================
Simplifie l'appel des services isolés depuis le code principal.

Usage:
    from isolated_services.client import ServiceClient
    
    client = ServiceClient('isolated_services/my_service.py')
    result = client.call({'param': 'value'}, timeout=60)
    print(result['output'])
"""
import subprocess
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path


class ServiceClient:
    """Client pour appeler un service isolé de façon élégante."""
    
    def __init__(self, service_path: str, python_executable: str = sys.executable):
        """
        Initialize client for a specific service.
        
        Args:
            service_path: Chemin vers le script du service
            python_executable: Interpréteur Python à utiliser
        """
        self.service_path = Path(service_path)
        self.python_executable = python_executable
        
        if not self.service_path.exists():
            raise FileNotFoundError(f"Service not found: {service_path}")
    
    def call(
        self,
        params: Dict[str, Any],
        timeout: Optional[int] = None,
        check_success: bool = True
    ) -> Dict[str, Any]:
        """
        Appeler le service avec les paramètres donnés.
        
        Args:
            params: Paramètres à envoyer au service
            timeout: Timeout en secondes (None = pas de limite)
            check_success: Raise exception si success=False
            
        Returns:
            Résultat JSON du service
            
        Raises:
            subprocess.TimeoutExpired: Si timeout dépassé
            RuntimeError: Si service échoue (returncode != 0)
            ValueError: Si success=False et check_success=True
        """
        # Convertir params en JSON
        params_json = json.dumps(params)
        
        # Appeler le service
        result = subprocess.run(
            [self.python_executable, str(self.service_path)],
            input=params_json,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Vérifier le code de retour
        if result.returncode != 0:
            # Essayer de parser la sortie pour obtenir l'erreur
            try:
                error_data = json.loads(result.stdout.strip().split('\n')[-1])
                error_msg = error_data.get('error', 'Unknown error')
                error_type = error_data.get('error_type', 'ServiceError')
            except:
                error_msg = result.stderr[-500:] if result.stderr else "Service failed"
                error_type = "ServiceError"
            
            raise RuntimeError(
                f"Service failed ({error_type}): {error_msg}\n"
                f"Return code: {result.returncode}"
            )
        
        # Parser la sortie JSON
        try:
            output = json.loads(result.stdout.strip().split('\n')[-1])
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON output from service: {e}\n"
                f"Output: {result.stdout[-500:]}"
            )
        
        # Vérifier le succès
        if check_success and not output.get('success', True):
            error = output.get('error', 'Unknown error')
            error_type = output.get('error_type', 'ServiceError')
            raise ValueError(f"Service returned error ({error_type}): {error}")
        
        return output
    
    def call_async(self, params: Dict[str, Any]) -> subprocess.Popen:
        """
        Appeler le service de façon asynchrone.
        
        Returns:
            subprocess.Popen object (use .communicate() to get result)
        """
        params_json = json.dumps(params)
        
        return subprocess.Popen(
            [self.python_executable, str(self.service_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )


class ServicePool:
    """Pool de services pour appels parallèles (future feature)."""
    
    def __init__(self, service_path: str, pool_size: int = 4):
        """
        TODO: Implémenter pool de workers pour parallélisme.
        
        Idea: Lancer N processus du service qui restent en vie
        et traitent les requêtes via queue (multiprocessing.Queue)
        """
        raise NotImplementedError("ServicePool not yet implemented")


# =====================================
# HELPERS POUR SERVICES COURANTS
# =====================================

def call_animation_service(
    prompt: str,
    num_keyframes: int = 5,
    width: int = 512,
    height: int = 512,
    **kwargs
) -> Dict[str, Any]:
    """
    Helper pour appeler le service d'animation facilement.
    
    Example:
        result = call_animation_service("cat walking", num_keyframes=3)
        keyframes_b64 = result['keyframes']
    """
    client = ServiceClient('isolated_services/animation_keyframes.py')
    
    params = {
        'prompt': prompt,
        'num_keyframes': num_keyframes,
        'width': width,
        'height': height,
        **kwargs
    }
    
    return client.call(params, timeout=600)


if __name__ == "__main__":
    # Test du client
    print("Testing ServiceClient...")
    
    # Test avec service d'exemple
    client = ServiceClient('isolated_services/base.py')
    result = client.call({'name': 'World'})
    
    print(f"✅ Result: {result}")
    print(f"Duration: {result['_meta']['duration_seconds']:.3f}s")
