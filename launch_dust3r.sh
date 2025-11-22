#!/bin/bash

###############################################################################
# Dust3r Launch Script - Configuration et lancement de DUSt3R
###############################################################################

set -e

# Configuration
DUST3R_DIR="/home/belikan/lifemodo-lab/dust3r"
# HF_TOKEN should be set in environment or .env file

echo "🚀 Configuration de DUSt3R..."
echo ""

# Ajouter Dust3r au PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$DUST3R_DIR"
export HF_TOKEN="$HF_TOKEN"

# Vérifier que les dépendances sont installées
echo "📦 Vérification des dépendances..."
python3 -c "
try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('✅ CUDA:', torch.version.cuda if torch.cuda.is_available() else 'N/A')

    import dust3r
    print('✅ Dust3r importé avec succès')

    from dust3r.model import AsymmetricCroCo3DStereo
    print('✅ Modèle Dust3r disponible')

except ImportError as e:
    print('❌ Erreur d\'import:', e)
    exit(1)
"

echo ""
echo "🎯 Lancement de l'application Dust3r..."

# Lancer l'application
cd /home/belikan/lifemodo-lab
exec streamlit run Dust3r/Dust3r.py --server.port=8530 --server.address=0.0.0.0