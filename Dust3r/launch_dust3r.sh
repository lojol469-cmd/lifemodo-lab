#!/bin/bash

# Script de lancement DUSt3R pour LifeModo AI Lab
# Ce script configure l'environnement et lance DUSt3R sur le port 8530

echo "🎯 Démarrage de DUSt3R Photogrammetry..."
echo "📍 Répertoire: $(pwd)"
echo "🔧 Configuration de l'environnement..."

# Aller dans le répertoire Dust3r
cd /home/belikan/lifemodo-lab/Dust3r

# Activer l'environnement conda si nécessaire
# conda activate lifemodo 2>/dev/null || echo "Conda non disponible, utilisation Python système"

# Configuration des variables d'environnement
export PYTHONPATH="/home/belikan/lifemodo-lab/Dust3r:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0  # Utiliser GPU 0 par défaut

# Vérifier que le modèle existe
MODEL_PATH="/home/belikan/lifemodo-lab/Dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Modèle DUSt3R non trouvé: $MODEL_PATH"
    echo "💡 Téléchargez d'abord le modèle avec download_model.py"
    exit 1
fi

echo "✅ Modèle trouvé: $MODEL_PATH"

# Vérifier que les dépendances sont installées
python -c "import torch; import torchvision; import dust3r; print('✅ Dépendances OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dépendances manquantes. Installez dust3r et torch."
    exit 1
fi

echo "🚀 Lancement de DUSt3R sur le port 8530..."
echo "🌐 Interface accessible sur: http://localhost:8530"
echo "⏱️ Chargement du modèle ViT-Large (~2-3 minutes)..."

# Lancer DUSt3R
python Dust3r/Dust3r.py --port 8530 --host 0.0.0.0

echo "🛑 DUSt3R arrêté."