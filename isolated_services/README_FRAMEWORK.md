# Isolated Services Framework

Framework Python pour créer des **services isolés** permettant d'éviter les conflits de dépendances entre bibliothèques incompatibles.

## Problème résolu

Certaines bibliothèques Python ont des dépendances conflictuelles. Par exemple:
- `transformers 4.46.0` et `diffusers` avec AnimateDiff
- Différentes versions de PyTorch/TensorFlow
- Conflits entre protobuf, numpy, etc.

Ce framework permet d'exécuter du code dans des processus séparés avec leurs propres environnements.

## Installation

```bash
# Développement
cd lifemodo-lab
pip install -e .

# Production
pip install isolated-services
```

## Utilisation

### 1. Créer un service

Créez un fichier `my_service.py`:

```python
from isolated_services import ServiceBase

class MyService(ServiceBase):
    def process(self, params):
        # Votre logique ici
        name = params.get("name", "World")
        return {"message": f"Hello {name}!"}

if __name__ == "__main__":
    service = MyService()
    service.run()
```

### 2. Appeler le service

```python
from isolated_services import IsolatedService

# Créer le client
service = IsolatedService(
    name="my_service",
    script_path="my_service.py"
)

# Appeler
result = service.call({"name": "Bob"})
print(result)  # {"message": "Hello Bob!"}
```

## Exemple AnimateDiff

Le framework inclut un exemple complet avec AnimateDiff pour la génération vidéo:

```python
from isolated_services import create_service

# Service AnimateDiff dans isol/animatediff_service.py
animatediff = create_service(
    name="AnimateDiff",
    script_path="isol/animatediff_service.py"
)

# Générer des frames
result = animatediff.call({
    "prompt": "a dog running",
    "negative_prompt": "bad quality",
    "num_frames": 16,
    "seed": 42
})

frames = result["frames"]  # Liste d'images base64
```

## Avantages

✅ **Isolation complète** - Chaque service tourne dans son propre processus  
✅ **Pas de conflits** - Les dépendances ne se mélangent jamais  
✅ **Simple** - API intuitive basée sur JSON  
✅ **Léger** - Utilise seulement la stdlib Python  
✅ **Flexible** - Timeout, gestion d'erreurs, etc.

## Structure

```
isolated_services/
├── __init__.py       # Point d'entrée
├── base.py          # ServiceBase - classe de base
├── service.py       # IsolatedService - client
└── README_FRAMEWORK.md

setup.py             # Installation pip
```

## Licence

MIT License - Libre d'utilisation

## Auteur

LifeModo Lab - Intelligence Artificielle Multimodale
