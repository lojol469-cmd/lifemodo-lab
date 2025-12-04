# 🧠 Guide d'Entraînement LLM - LifeModo AI Lab

## 🎯 **Objectif**
Créer des **LLMs spécialisés par PDF** avec fine-tuning LoRA et export multi-formats.

---

## 📋 **Fonctionnalités**

### ✅ **Mode Séparé LLM**
- **1 LLM par PDF** : Chaque document a son propre modèle expert
- **Fine-tuning LoRA** : Adaptation légère (16 rangs) du modèle de base
- **Quantization 4-bit** : Économie de RAM (2-3 GB par modèle)
- **Export automatique** : ONNX, Safetensors, HuggingFace

---

## 🚀 **Utilisation**

### **1. Importer vos PDFs**
```
📁 Importation Données
└─ Upload PDFs → Lifemodo Lab extrait le texte automatiquement
```

### **2. Sélectionner "Langage (Transformers)"**
```
🧠 Entraînement IA
├─ Modèles : ☑️ Langage (Transformers)
└─ Le système détecte les PDFs et propose le mode séparé
```

### **3. Configurer l'entraînement**
- **Modèle de base** : `microsoft/phi-2` (2.7B params, rapide)
  - Alternative : `meta-llama/Llama-3.2-1B`, `mistralai/Mistral-7B-v0.1`
- **Époques** : 3-5 (plus = meilleure spécialisation)
- **LoRA Rank** : 16 (équilibre qualité/vitesse)

### **4. Lancer l'entraînement**
```
🚀 Lancer entraînement
└─ Pour chaque PDF :
    ├─ Extraction du texte (si non fait)
    ├─ Fine-tuning LoRA (adapte le modèle au contenu)
    ├─ Sauvegarde du modèle (models/llm_{pdf_name}/)
    └─ Export ONNX automatique (exports/llm_{pdf_name}.onnx)
```

---

## 📦 **Formats d'Export**

### **1. HuggingFace Format (Défaut)**
```bash
models/llm_Guide_Word_2013/
├── adapter_config.json       # Config LoRA
├── adapter_model.safetensors # Poids LoRA
├── tokenizer_config.json
└── tokenizer.model
```

**Utilisation :**
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("models/llm_Guide_Word_2013")
tokenizer = AutoTokenizer.from_pretrained("models/llm_Guide_Word_2013")

prompt = "Comment insérer un tableau dans Word 2013 ?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

### **2. ONNX (Production)**
```bash
exports/llm_Guide_Word_2013.onnx  # 5-10 GB
```

**Utilisation avec ONNX Runtime :**
```python
import onnxruntime as ort

session = ort.InferenceSession("exports/llm_Guide_Word_2013.onnx")
# Inférence rapide cross-platform
```

### **3. Safetensors (Sécurisé)**
```bash
models/llm_Guide_Word_2013/adapter_model.safetensors
```
- Format sûr (pas d'exécution de code arbitraire)
- Compatible HuggingFace Hub

---

## 🎓 **Modèles de Base Recommandés**

| Modèle | Taille | RAM Min | Vitesse | Cas d'usage |
|--------|--------|---------|---------|-------------|
| `microsoft/phi-2` | 2.7B | 6 GB | ⚡⚡⚡ | Idéal pour débuter |
| `Qwen/Qwen2.5-1.5B` | 1.5B | 4 GB | ⚡⚡⚡⚡ | Ultra-rapide |
| `meta-llama/Llama-3.2-1B` | 1B | 3 GB | ⚡⚡⚡⚡ | Léger, efficient |
| `mistralai/Mistral-7B-v0.3` | 7B | 16 GB | ⚡⚡ | Qualité supérieure |

---

## 💡 **Cas d'Usage**

### **1. Assistant Documentation**
```python
# LLM entraîné sur "Guide Word 2013"
prompt = "Comment créer un sommaire automatique ?"
# → Réponse basée sur le PDF, pas hallucinations génériques
```

### **2. Chatbot Technique**
```python
# LLM entraîné sur manuels techniques
prompt = "Erreur #0x80070005 lors de l'installation"
# → Diagnostics précis du manuel
```

### **3. Générateur de Contenu**
```python
# LLM entraîné sur corpus marketing
prompt = "Rédige une description produit pour [...]"
# → Style cohérent avec la marque
```

---

## ⚙️ **Configuration Avancée**

### **LoRA Hyperparamètres**
```python
LoraConfig(
    r=16,              # Rank : 8-64 (↑ = plus de capacité)
    lora_alpha=32,     # Scaling : généralement 2*r
    lora_dropout=0.05, # Régularisation
    target_modules=["q_proj", "v_proj"]  # Attention layers
)
```

### **Optimisation Mémoire**
- **4-bit quantization** : Réduit RAM de 75%
- **Gradient accumulation** : Steps=4 simule batch_size=8
- **FP16 training** : 2x plus rapide

---

## 🐛 **Troubleshooting**

### **Erreur : "CUDA out of memory"**
```python
# Réduire batch_size
per_device_train_batch_size=1
gradient_accumulation_steps=8
```

### **Erreur : "ImportError: peft not found"**
```bash
pip install peft optimum accelerate bitsandbytes
```

### **Modèle trop lent**
- Utiliser `microsoft/phi-2` au lieu de Mistral-7B
- Activer `load_in_4bit=True`
- Réduire `max_length=256`

---

## 📊 **Métriques d'Entraînement**

Pendant l'entraînement, surveillez :
- **Loss** : Doit diminuer (< 1.0 = bon)
- **RAM Usage** : Stable sans pics
- **GPU Utilization** : 70-90% optimal

---

## 🚀 **Déploiement**

### **Option 1 : API FastAPI**
```python
from fastapi import FastAPI
from peft import AutoPeftModelForCausalLM

app = FastAPI()
model = AutoPeftModelForCausalLM.from_pretrained("models/llm_Guide_Word_2013")

@app.post("/generate")
def generate(prompt: str):
    outputs = model.generate(tokenizer(prompt, return_tensors="pt").input_ids)
    return {"response": tokenizer.decode(outputs[0])}
```

### **Option 2 : ONNX Runtime**
```python
import onnxruntime as ort
session = ort.InferenceSession("exports/llm_Guide_Word_2013.onnx")
# 3-5x plus rapide que PyTorch !
```

### **Option 3 : HuggingFace Spaces**
```bash
git lfs install
git clone https://huggingface.co/spaces/YOUR_USERNAME/word-assistant
cp -r models/llm_Guide_Word_2013/* word-assistant/
cd word-assistant && git add . && git commit -m "Add model" && git push
```

---

## 🎯 **Prochaines Étapes**

1. ✅ Entraîner votre premier LLM sur un PDF
2. ✅ Tester avec des prompts réels
3. ✅ Exporter en ONNX pour déploiement
4. 🚀 Publier sur HuggingFace Hub
5. 🌐 Créer une API REST avec FastAPI

---

## 📚 **Ressources**

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Méthode PEFT
- [Phi-2 Model Card](https://huggingface.co/microsoft/phi-2)
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)

---

**🎉 Vous êtes maintenant prêt à créer des LLMs experts sur vos propres documents !**
