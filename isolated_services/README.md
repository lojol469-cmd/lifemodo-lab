# 🚀 Isolated Services Framework

**Pattern**: Process isolation + JSON-RPC over stdio  
**Use case**: Éviter les conflits de dépendances entre packages Python incompatibles

## 🎯 Pourquoi ce framework ?

### Le problème

```python
# ❌ Ça plante à cause de conflits de dépendances
from transformers import pipeline  # veut peft==0.18.0
from diffusers import StableDiffusionPipeline  # veut peft>=0.17.0 mais <0.18.0
# ImportError: cannot import name 'MODELS_TO_PIPELINE'
```

### La solution

```python
# ✅ Aucun conflit ! Chaque service = subprocess indépendant
from isolated_services.client import ServiceClient

client = ServiceClient('isolated_services/my_service.py')
result = client.call({'input': 'data'})
```

## 📐 Architecture

```
┌─────────────────────────────────────────┐
│  Application principale (Streamlit)     │
│  ├─ transformers==4.57.3               │
│  ├─ torch==2.5.0                       │
│  └─ NO diffusers import                │
└─────────────────────────────────────────┘
                  │
                  │ subprocess + JSON
                  ▼
┌─────────────────────────────────────────┐
│  Service isolé (animation_keyframes)    │
│  ├─ diffusers==0.35.2                  │
│  ├─ peft==0.18.0                       │
│  └─ transformers==4.57.3               │
└─────────────────────────────────────────┘
```

**Avantages** :
- ✅ **0 conflits** : Chaque service a son propre espace mémoire
- ✅ **Simplicité** : Pas de Docker, Celery, Redis...
- ✅ **Performance** : Pas d'overhead réseau (local)
- ✅ **Debugging** : Logs stderr séparés, tracebacks complets
- ✅ **Portabilité** : Pure Python, fonctionne partout

## 🚀 Quick Start

### 1. Créer un service

```python
# isolated_services/my_service.py
from base import ServiceBase

class MyService(ServiceBase):
    def process(self, params):
        # Votre code ici
        name = params.get('name', 'World')
        return {'message': f'Hello {name}!'}

if __name__ == '__main__':
    MyService().run()
```

### 2. Appeler le service

#### Option A : Client helper (recommandé)

```python
from isolated_services.client import ServiceClient

client = ServiceClient('isolated_services/my_service.py')
result = client.call({'name': 'Alice'}, timeout=30)
print(result['message'])  # "Hello Alice!"
```

#### Option B : Subprocess manuel

```python
import subprocess, json, sys

params = {'name': 'Bob'}
result = subprocess.run(
    [sys.executable, 'isolated_services/my_service.py'],
    input=json.dumps(params),
    capture_output=True,
    text=True
)

output = json.loads(result.stdout.strip().split('\n')[-1])
print(output['message'])  # "Hello Bob!"
```

## 🎨 Exemple réel : Service d'animation

```python
# isolated_services/animation_keyframes.py
from base import ServiceBase
from diffusers import StableDiffusionPipeline  # Aucun conflit !
import torch

class AnimationService(ServiceBase):
    def __init__(self):
        super().__init__(log_level=logging.INFO)
        self.pipe = None
    
    def process(self, params):
        # Charger modèle
        if self.pipe is None:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            ).to("cuda")
        
        # Générer image
        prompt = params['prompt']
        image = self.pipe(prompt).images[0]
        
        # Encoder en base64
        import base64, io
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {'image': img_b64}

if __name__ == '__main__':
    AnimationService().run()
```

Utilisation :

```python
from isolated_services.client import call_animation_service

result = call_animation_service("cat walking", num_keyframes=5)
keyframes = result['keyframes']  # List[str] base64
```

## 📚 Features avancées

### Validation des entrées/sorties

```python
class MyService(ServiceBase):
    def validate_input(self, params):
        if 'required_field' not in params:
            raise ValueError("Missing required_field")
        if not isinstance(params['required_field'], int):
            raise ValueError("required_field must be int")
    
    def validate_output(self, result):
        if 'result' not in result:
            raise ValueError("Missing result field")
```

### Logging personnalisé

```python
# Service avec logs dans fichier
class MyService(ServiceBase):
    def __init__(self):
        super().__init__(
            log_level=logging.DEBUG,
            log_file='/tmp/my_service.log'
        )
    
    def process(self, params):
        self.logger.info("Processing started")
        self.logger.debug(f"Params: {params}")
        # ...
```

### Métriques automatiques

Chaque réponse contient automatiquement :

```python
{
    "success": true,
    "your_data": "...",
    "_meta": {
        "duration_seconds": 1.234,
        "timestamp": "2025-12-05T10:30:00"
    }
}
```

### Gestion d'erreurs robuste

```python
try:
    result = client.call({'bad': 'data'})
except RuntimeError as e:
    # Service a crashé
    print(f"Service error: {e}")
except ValueError as e:
    # Service a retourné success=False
    print(f"Business error: {e}")
```

## 🔥 Cas d'usage avancés

### 1. Multiple services en parallèle

```python
import concurrent.futures

def call_service(service, params):
    client = ServiceClient(f'isolated_services/{service}.py')
    return client.call(params)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(call_service, 'service1', {'id': 1}),
        executor.submit(call_service, 'service2', {'id': 2}),
        executor.submit(call_service, 'service3', {'id': 3}),
    ]
    results = [f.result() for f in futures]
```

### 2. Service avec GPU isolation

```python
# isolated_services/gpu_service.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force GPU 0

class GPUService(ServiceBase):
    def process(self, params):
        # Ce service utilise uniquement GPU 0
        pass
```

### 3. Service avec imports lourds (lazy loading)

```python
class HeavyService(ServiceBase):
    def __init__(self):
        super().__init__()
        self.model = None  # Pas encore chargé
    
    def process(self, params):
        # Charger uniquement au premier appel
        if self.model is None:
            self.logger.info("Loading heavy model...")
            import torch
            self.model = torch.load('huge_model.pt')
        
        return self.model.predict(params['input'])
```

## 🆚 Comparaison avec alternatives

| Solution | Overhead | Simplicité | Isolation | Notre framework |
|----------|----------|------------|-----------|----------------|
| **Docker** | Élevé (containers) | Moyenne | Complète | ❌ |
| **Celery** | Moyen (Redis/RabbitMQ) | Faible | Moyenne | ❌ |
| **Microservices HTTP** | Élevé (réseau) | Faible | Complète | ❌ |
| **venv séparés** | Faible | Moyenne | Partielle | ❌ |
| **Isolated Services** | Minimal (subprocess) | Élevée | Complète | ✅ |

## 🐛 Debugging

### Logs du service

```python
# Les logs vont vers stderr, pas stdout (pour ne pas polluer JSON)
client = ServiceClient('my_service.py')
result = client.call({'data': 'test'})

# Voir les logs :
# tail -f /tmp/my_service.log  (si log_file configuré)
```

### Mode debug

```python
class MyService(ServiceBase):
    def __init__(self):
        super().__init__(log_level=logging.DEBUG)  # Verbose
```

### Traceback complet

En cas d'erreur, le service retourne :

```json
{
    "success": false,
    "error": "ValueError: Invalid input",
    "error_type": "ValueError",
    "context": "Validation error",
    "traceback": "Traceback (most recent call last):\n  File ..."
}
```

## 📦 Installation

```bash
# Rien à installer ! Pure stdlib Python
# Optionnel : dépendances spécifiques dans chaque service
pip install diffusers torch transformers  # Pour animation_service
```

## 🎓 Best Practices

1. **Un service = Une responsabilité** (Unix philosophy)
2. **Logs vers stderr uniquement** (stdout = JSON uniquement)
3. **Timeout adapté** (GPU = 600s, CPU = 60s)
4. **Validation explicite** (fail fast)
5. **Métriques activées** (monitoring performance)

## 📄 License

MIT License - Do whatever you want with this framework!

## 🙏 Inspirations

- **Language Server Protocol (LSP)** : Microsoft
- **JSON-RPC 2.0** : Standard
- **Unix Philosophy** : "Do one thing well"
- **Microservices pattern** : Netflix, Amazon

## 🚀 Roadmap

- [ ] ServicePool (workers pool)
- [ ] Streaming support (chunked responses)
- [ ] Async/await API
- [ ] Service discovery
- [ ] Health checks
- [ ] Metrics dashboard

---

**Created by**: belikan  
**Date**: December 2025  
**Status**: Production-ready ✅
