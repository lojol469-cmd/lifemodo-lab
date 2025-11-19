#!/bin/bash
# 🚀 Lanceur du Serveur API Robotique Intelligent

echo "🤖 LANCEMENT DU SERVEUR API ROBOTIQUE 🤖"
echo "========================================"

# Vérification de l'environnement
if [ ! -d "/home/belikan/lifemodo_api" ]; then
    echo "❌ Répertoire lifemodo_api non trouvé"
    exit 1
fi

cd /home/belikan/lifemodo_api

# Vérification des dépendances
echo "📦 Vérification des dépendances..."
python -c "import fastapi, uvicorn, intelligent_robot" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Installation des dépendances manquantes..."
    pip install fastapi uvicorn python-multipart
fi

echo "✅ Dépendances vérifiées"

# Démarrage du serveur
echo "🚀 Démarrage du serveur sur http://localhost:8000"
echo "📚 Documentation API: http://localhost:8000/docs"
echo "🌐 Interface web: http://localhost:8000"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter le serveur"
echo ""

python robot_api_server.py