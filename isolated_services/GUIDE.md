# Guide d'installation et d'utilisation

## Installation rapide

### Mode développement

```bash
cd /home/belikan/lifemodo-lab
pip install -e .
```

### Mode production

```bash
pip install git+https://github.com/votre-repo/isolated-services.git
```

## Utilisation basique

### 1. Créer un service

Fichier `my_service.py`:

```python
from isolated_services import ServiceBase

class MyService(ServiceBase):
    def process(self, params):
        name = params.get("name", "World")
        return {"greeting": f"Hello {name}!"}

if __name__ == "__main__":
    service = MyService()
    service.run()
```

### 2. Utiliser le service

```python
from isolated_services import IsolatedService

service = IsolatedService("my_service", "my_service.py")
result = service.call({"name": "Alice"})
print(result["greeting"])  # "Hello Alice!"
```

## Exemples inclus

### Calculatrice (simple)

```bash
python examples/test_calculator.py
```

### AnimateDiff (IA générative)

```bash
python test_isolated_framework.py
```

## Cas d'usage

### Éviter les conflits de dépendances

```python
# Service A utilise transformers 4.46.0
# Service B utilise diffusers avec AnimateDiff
# = CONFLIT sans isolated_services

# Avec isolated_services : PAS DE PROBLÈME
service_a = IsolatedService("transformers", "service_a.py")
service_b = IsolatedService("animatediff", "service_b.py")

result_a = service_a.call({"text": "..."})
result_b = service_b.call({"prompt": "..."})
```

### Timeout et gestion d'erreurs

```python
try:
    result = service.call(params, timeout=60)  # 60 secondes max
except TimeoutError:
    print("Service trop lent")
except RuntimeError as e:
    print(f"Erreur service: {e}")
```

## API Référence

### ServiceBase

Classe de base pour créer un service.

**Méthode à implémenter:**

```python
def process(self, params: dict) -> dict:
    """
    Args:
        params: Paramètres d'entrée (JSON-sérialisable)
    Returns:
        Résultat (JSON-sérialisable)
    """
```

### IsolatedService

Client pour appeler un service.

**Constructeur:**

```python
IsolatedService(
    name: str,              # Nom du service
    script_path: str,       # Chemin vers le script
    python_executable: str  # Optionnel: python à utiliser
)
```

**Méthode call:**

```python
service.call(
    params: dict,      # Paramètres
    timeout: int       # Timeout en secondes (optionnel)
) -> dict
```

## Architecture

```
isolated_services/
├── __init__.py           # Exports publics
├── base.py              # ServiceBase
├── service.py           # IsolatedService
└── README_FRAMEWORK.md  # Cette doc

examples/
├── calculator_service.py    # Service simple
└── test_calculator.py       # Test

isol/
└── animatediff_service_v2.py  # Service IA complexe

setup.py                 # Installation pip
```

## Dépannage

### Le service ne démarre pas

- Vérifier que le script existe
- Vérifier les permissions d'exécution
- Vérifier l'import de `ServiceBase`

### Erreur JSON

- S'assurer que `process()` retourne un dict
- Éviter les objets non-sérialisables (numpy arrays, PIL images)
- Utiliser base64 pour les images

### Timeout

- Augmenter le timeout: `service.call(params, timeout=300)`
- Optimiser le service (GPU, quantization, etc.)

## Contribuer

1. Fork le repo
2. Créer une branche feature
3. Ajouter tests
4. Pull request

## Support

GitHub Issues: https://github.com/votre-repo/issues
