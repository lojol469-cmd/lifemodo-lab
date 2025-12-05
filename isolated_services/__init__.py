"""
Isolated Services Framework
===========================

Framework pour créer des services isolés permettant d'éviter les conflits
de dépendances entre bibliothèques incompatibles.

Usage:
    from isolated_services import IsolatedService
    
    # Créer un service
    service = IsolatedService(
        name="my_service",
        script_path="path/to/service.py"
    )
    
    # Appeler le service
    result = service.call({"param": "value"})
"""

from .service import IsolatedService
from .base import ServiceBase

__version__ = "0.1.0"
__all__ = ["IsolatedService", "ServiceBase"]
