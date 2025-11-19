#!/usr/bin/env python3
"""
🤖 SERVEUR API ROBOTIQUE INTELLIGENT 🤖
Serveur FastAPI pour l'accès aux robots spécialisés par domaine
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import base64
import io

# Import du système robotique
from intelligent_robot import IntelligentRobot, initialize_robot_system

# Configuration du serveur
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "🤖 API Robotique Intelligent"
API_DESCRIPTION = "API pour accéder aux robots spécialisés par domaine"
API_VERSION = "1.0.0"

# Configuration sécurité
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "robot2025")
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"

# Initialisation de l'application FastAPI
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs" if not ENABLE_AUTH else None,
    redoc_url="/redoc" if not ENABLE_AUTH else None
)

# Sécurité
security = HTTPBasic()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable globale pour le système robotique
robot_system: Optional[IntelligentRobot] = None

# Métriques d'utilisation
api_metrics = {
    "requests_total": 0,
    "requests_by_domain": {},
    "errors_total": 0,
    "start_time": datetime.now().isoformat(),
    "active_users": set(),
    "last_request_time": None
}

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Vérification des credentials si auth activée"""
    if not ENABLE_AUTH:
        return True

    correct_username = secrets.compare_digest(credentials.username, API_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, API_PASSWORD)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants incorrects",
            headers={"WWW-Authenticate": "Basic"},
        )

    # Track active users
    api_metrics["active_users"].add(credentials.username)
    api_metrics["last_request_time"] = datetime.now().isoformat()

    return True

@app.on_event("startup")
async def startup_event():
    """Initialisation du système au démarrage"""
    global robot_system
    print("🚀 Démarrage du serveur API robotique...")

    try:
        # Initialisation du système robotique
        robot_system = initialize_robot_system()
        print("✅ Système robotique initialisé avec succès")

        # Chargement du cerveau Mistral si disponible
        if hasattr(robot_system, 'load_brain'):
            try:
                robot_system.load_brain()
                print("🧠 Cerveau Mistral chargé")
            except Exception as e:
                print(f"⚠️ Cerveau Mistral non disponible: {e}")

        # Vérification des dépendances critiques
        check_critical_dependencies()

    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        robot_system = None

def check_critical_dependencies():
    """Vérifie les dépendances critiques au démarrage"""
    critical_deps = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("ultralytics", "Ultralytics YOLO"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn")
    ]

    missing_deps = []
    for module, name in critical_deps:
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(name)

    if missing_deps:
        print(f"⚠️ Dépendances manquantes: {', '.join(missing_deps)}")
        print("Installez avec: pip install torch transformers ultralytics fastapi uvicorn")
    else:
        print("✅ Toutes les dépendances critiques sont installées")

@app.on_event("startup")
async def startup_event():
    """Initialisation du système au démarrage"""
    global robot_system
    print("🚀 Démarrage du serveur API robotique...")

    try:
        # Initialisation du système robotique
        robot_system = initialize_robot_system()
        print("✅ Système robotique initialisé avec succès")

        # Chargement du cerveau Mistral si disponible
        if hasattr(robot_system, 'load_brain'):
            try:
                robot_system.load_brain()
                print("🧠 Cerveau Mistral chargé")
            except Exception as e:
                print(f"⚠️ Cerveau Mistral non disponible: {e}")

    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        robot_system = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil avec interface web"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{API_TITLE}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                padding: 50px 0;
            }}
            .header h1 {{
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .header p {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .domains {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }}
            .domain-card {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: transform 0.3s ease;
            }}
            .domain-card:hover {{
                transform: translateY(-5px);
            }}
            .domain-card h3 {{
                font-size: 1.5em;
                margin-bottom: 15px;
            }}
            .domain-card .icon {{
                font-size: 3em;
                margin-bottom: 15px;
            }}
            .domain-card .models {{
                font-size: 0.9em;
                opacity: 0.8;
                margin-top: 15px;
            }}
            .api-info {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 20px;
                margin: 30px 0;
                text-align: center;
            }}
            .api-info h2 {{
                margin-bottom: 15px;
            }}
            .links {{
                display: flex;
                justify-content: center;
                gap: 20px;
                margin: 30px 0;
            }}
            .btn {{
                display: inline-block;
                padding: 12px 24px;
                background: rgba(255, 255, 255, 0.2);
                color: white;
                text-decoration: none;
                border-radius: 25px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                transition: all 0.3s ease;
            }}
            .btn:hover {{
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-2px);
            }}
            .status {{
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(0, 255, 0, 0.8);
                color: white;
                padding: 10px 15px;
                border-radius: 20px;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="status" id="status">
            🔄 Chargement...
        </div>
        <div class="container">
            <div class="header">
                <h1>🤖 {API_TITLE}</h1>
                <p>Plateforme d'IA robotique spécialisée par domaine</p>
            </div>

            <div class="domains" id="domains">
                <!-- Les domaines seront chargés dynamiquement -->
            </div>

            <div class="api-info">
                <h2>📡 APIs Disponibles</h2>
                <p>Utilisez les endpoints suivants pour accéder aux robots spécialisés :</p>
                <div id="apis">
                    <!-- Les APIs seront chargées dynamiquement -->
                </div>
            </div>

            <div class="links">
                <a href="/docs" class="btn">📚 Documentation API</a>
                <a href="/metrics" class="btn">📊 Métriques</a>
                <a href="/health" class="btn">❤️ Santé</a>
            </div>
        </div>

        <script>
            async function loadSystemInfo() {{
                try {{
                    const response = await fetch('/system/info');
                    const data = await response.json();

                    // Mise à jour du statut
                    const status = document.getElementById('status');
                    status.textContent = data.status === 'ready' ? '✅ Système prêt' : '⚠️ Système en cours d\\'initialisation';

                    // Chargement des domaines
                    const domainsContainer = document.getElementById('domains');
                    domainsContainer.innerHTML = '';

                    for (const [domain, models] of Object.entries(data.domains)) {{
                        const card = document.createElement('div');
                        card.className = 'domain-card';

                        const icon = getDomainIcon(domain);
                        const modelList = models.join(', ');

                        card.innerHTML = `
                            <div class="icon">${{icon}}</div>
                            <h3>${{domain.charAt(0).toUpperCase() + domain.slice(1)}}</h3>
                            <p>Modèles spécialisés pour ce domaine</p>
                            <div class="models">📦 ${{modelList}}</div>
                        `;

                        domainsContainer.appendChild(card);
                    }}

                    // Chargement des APIs
                    const apisContainer = document.getElementById('apis');
                    apisContainer.innerHTML = '';

                    for (const [model, api] of Object.entries(data.apis)) {{
                        const apiDiv = document.createElement('div');
                        apiDiv.style.margin = '10px 0';
                        apiDiv.innerHTML = `<code>POST ${{api.endpoint}}</code> - ${{model}}`;
                        apisContainer.appendChild(apiDiv);
                    }}

                }} catch (error) {{
                    console.error('Erreur de chargement:', error);
                    document.getElementById('status').textContent = '❌ Erreur de chargement';
                }}
            }}

            function getDomainIcon(domain) {{
                const icons = {{
                    'vision': '👁️',
                    'language': '💬',
                    'audio': '🔊',
                    'robotics': '🤖'
                }};
                return icons[domain] || '🔧';
            }}

            // Chargement initial
            loadSystemInfo();

            // Rechargement périodique
            setInterval(loadSystemInfo, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check(authenticated: bool = Depends(verify_credentials)):
    """Vérification de santé du système"""
    global robot_system

    health_status = {
        "status": "healthy" if robot_system else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "robot_system": "initialized" if robot_system else "not_initialized",
        "brain_loaded": robot_system.has_brain if robot_system else False,
        "models_count": len(robot_system.models) if robot_system else 0,
        "domains": list(robot_system.models.keys()) if robot_system else [],
        "auth_enabled": ENABLE_AUTH,
        "active_users": len(api_metrics["active_users"])
    }

    status_code = 200 if robot_system else 503
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/metrics")
async def get_metrics(authenticated: bool = Depends(verify_credentials)):
    """Métriques d'utilisation de l'API"""
    global api_metrics

    current_time = datetime.now()
    uptime = str(current_time - datetime.fromisoformat(api_metrics["start_time"]))

    metrics = {
        "uptime": uptime,
        "requests_total": api_metrics["requests_total"],
        "requests_by_domain": api_metrics["requests_by_domain"],
        "errors_total": api_metrics["errors_total"],
        "active_users": len(api_metrics["active_users"]),
        "start_time": api_metrics["start_time"],
        "last_request_time": api_metrics["last_request_time"],
        "current_time": current_time.isoformat()
    }

    return JSONResponse(content=metrics)

@app.get("/system/info")
async def system_info():
    """Informations sur le système robotique"""
    global robot_system

    if not robot_system:
        raise HTTPException(status_code=503, detail="Système robotique non initialisé")

    info = {
        "status": "ready",
        "brain_loaded": robot_system.has_brain,
        "domains": {},
        "apis": {},
        "datasets": list(robot_system.datasets.keys()) if robot_system.datasets else []
    }

    # Collecte des informations par domaine
    for domain, models in robot_system.models.items():
        info["domains"][domain] = list(models.keys())

        # APIs pour ce domaine
        for model_name in models.keys():
            api_info = robot_system.apis.get(model_name, {})
            if api_info:
                info["apis"][model_name] = {
                    "endpoint": api_info.get("endpoint", ""),
                    "domain": domain
                }

    return JSONResponse(content=info)

@app.post("/api/vision/infer")
async def vision_inference(
    file: UploadFile = File(...),
    model: str = "vision_yolo_trained",
    task: str = "detect"
):
    """API d'inférence pour la vision"""
    return await _handle_inference("vision", model, file, task)

@app.post("/api/language/infer")
async def language_inference(
    text: str,
    model: str = "language_transformers",
    task: str = "analyze"
):
    """API d'inférence pour le langage"""
    return await _handle_inference("language", model, None, task, text_input=text)

@app.post("/api/audio/infer")
async def audio_inference(
    file: UploadFile = File(...),
    model: str = "audio_pytorch",
    task: str = "transcribe"
):
    """API d'inférence pour l'audio"""
    return await _handle_inference("audio", model, file, task)

@app.post("/api/robotics/infer")
async def robotics_inference(
    file: UploadFile = File(...),
    model: str = "robotics_aloha_cube",
    task: str = "predict_action"
):
    """API d'inférence pour la robotique"""
    return await _handle_inference("robotics", model, file, task)

async def _handle_inference(
    domain: str,
    model_name: str,
    file: Optional[UploadFile] = None,
    task: str = "",
    text_input: str = ""
):
    """Gestionnaire générique d'inférence"""
    global robot_system, api_metrics

    # Mise à jour des métriques
    api_metrics["requests_total"] += 1
    if domain not in api_metrics["requests_by_domain"]:
        api_metrics["requests_by_domain"][domain] = 0
    api_metrics["requests_by_domain"][domain] += 1

    try:
        if not robot_system:
            raise HTTPException(status_code=503, detail="Système robotique non disponible")

        # Vérification du domaine et modèle
        if domain not in robot_system.models or model_name not in robot_system.models[domain]:
            raise HTTPException(
                status_code=404,
                detail=f"Modèle '{model_name}' non trouvé dans le domaine '{domain}'"
            )

        # Préparation des données d'entrée
        input_data = {}

        if text_input:
            input_data["text"] = text_input
        elif file:
            # Lecture du fichier
            content = await file.read()
            input_data["file_content"] = content
            input_data["filename"] = file.filename
            input_data["content_type"] = file.content_type

            # Encodage base64 pour la réponse
            input_data["file_base64"] = base64.b64encode(content).decode('utf-8')

        # Exécution de l'inférence
        result = await robot_system.think_and_decide(task, domain, model_name, input_data)

        response = {
            "success": True,
            "domain": domain,
            "model": model_name,
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }

        return JSONResponse(content=response)

    except HTTPException:
        api_metrics["errors_total"] += 1
        raise
    except Exception as e:
        api_metrics["errors_total"] += 1
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence: {str(e)}")

@app.post("/robot/analyze")
async def analyze_task(task_description: str):
    """Analyse intelligente d'une tâche et recommandation de robot"""
    global robot_system

    if not robot_system:
        raise HTTPException(status_code=503, detail="Système robotique non disponible")

    try:
        # Analyse de la tâche
        analysis = robot_system.think_and_decide(task_description, "general", "brain", {})

        response = {
            "task": task_description,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse: {str(e)}")

if __name__ == "__main__":
    print(f"🚀 Démarrage du serveur {API_TITLE} sur {API_HOST}:{API_PORT}")
    print("📚 Documentation disponible sur: http://localhost:8000/docs")
    print("🌐 Interface web disponible sur: http://localhost:8000")
    uvicorn.run(
        "robot_api_server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )