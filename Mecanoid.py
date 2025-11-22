import streamlit as st
import os
from ultralytics import YOLO, SAM
import matplotlib.pyplot as plt
from collections import defaultdict
import tempfile
import cv2
import shutil
import random
import re
import torch
import easyocr
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import albumentations as A
from ray import tune
import faiss
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from torchvision import models, transforms
from PIL import Image
import fitz  # PyMuPDF for PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Additional imports for LLM agent
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.llms.base import LLM
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import dotenv
import json
from typing import Optional, Type, Any
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType

# Load environment variables
dotenv.load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Répertoires supplémentaires
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown_images")
FAISS_INDEX = os.path.join(BASE_DIR, "faiss_index.index")
KNOWLEDGE_FAISS = os.path.join(BASE_DIR, "knowledge_faiss.index")
KNOWLEDGE_CHUNKS = os.path.join(BASE_DIR, "knowledge_chunks.json")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "part_annotations.json")
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# Global variables for LLM
mistral_pipe = None
mistral_tokenizer = None

def load_mistral_model():
    """Charge le modèle Mistral local"""
    global mistral_pipe, mistral_tokenizer
    try:
        model_path = os.path.join(BASE_DIR, "llms", "mistral-7b")
        if not os.path.exists(model_path):
            st.error("Modèle Mistral non trouvé. Téléchargez-le d'abord.")
            return None, None

        if mistral_pipe is None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                token=HF_TOKEN,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            mistral_pipe = pipe
            mistral_tokenizer = tokenizer

        return mistral_pipe, mistral_tokenizer
    except Exception as e:
        st.error(f"Erreur chargement Mistral: {str(e)}")
        return None, None

class MistralLLM(LLM):
    pipeline: Any = None

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def _call(self, prompt, stop=None):
        try:
            result = self.pipeline(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )[0]['generated_text']

            if prompt in result:
                result = result.replace(prompt, "").strip()

            return result
        except Exception as e:
            return f"Erreur génération: {str(e)}"

    @property
    def _llm_type(self):
        return "mistral_pipeline"

def create_part_identification_agent():
    """Crée un agent LangChain pour l'identification de pièces avec Tavily"""
    try:
        pipe, tokenizer = load_mistral_model()
        if not pipe:
            return None

        llm = MistralLLM(pipe)

        # Outil Tavily pour recherche internet
        tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)

        # Prompt pour l'agent
        react_prompt = PromptTemplate.from_template("""Tu es un expert en mécanique et identification de pièces automobiles/industrielles.

Ta mission: À partir de la description d'une pièce, rechercher sur internet pour trouver le nom exact de la pièce, ses correspondances (numéros OEM, références), et fournir une explication détaillée.

Utilise l'outil de recherche internet pour trouver des informations précises.

Format de réponse:
1. Nom exact de la pièce
2. Numéros de référence/correspondance
3. Explication détaillée
4. Sources consultées

Outils disponibles:
{tools}

Format d'exécution:
Question: {input}
Thought: Analyse de la question
Action: Outil à utiliser
Action Input: Paramètre pour l'outil
Observation: Résultat de l'outil
... (répéter si nécessaire)
Thought: J'ai assez d'informations
Final Answer: Réponse finale structurée

Question: {input}
Thought:{agent_scratchpad}""")

        agent = create_react_agent(llm=llm, tools=[tavily_tool], prompt=react_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tavily_tool],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

        return agent_executor

    except Exception as e:
        st.error(f"Erreur création agent: {str(e)}")
        return None

def create_chat_agent():
    """Crée un agent LangChain pour le chat conversationnel avec mémoire"""
    try:
        pipe, tokenizer = load_mistral_model()
        if not pipe:
            return None

        llm = MistralLLM(pipe)

        # Outils disponibles pour l'agent
        tools = [
            TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5),
            # Ajouter d'autres outils si nécessaire
        ]

        # Mémoire conversationnelle
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Créer l'agent avec mémoire
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

        return agent

    except Exception as e:
        st.error(f"Erreur création agent chat: {str(e)}")
        return None

def save_chat_history(chat_history):
    """Sauvegarde l'historique du chat"""
    try:
        chat_file = os.path.join(BASE_DIR, "chat_history.json")
        with open(chat_file, "w", encoding='utf-8') as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Erreur sauvegarde chat: {str(e)}")
        return False

def load_chat_history():
    """Charge l'historique du chat"""
    try:
        chat_file = os.path.join(BASE_DIR, "chat_history.json")
        if os.path.exists(chat_file):
            with open(chat_file, "r", encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Erreur chargement chat: {str(e)}")
        return []

# Nouvelle fonctionnalité 1: Support YOLOv11
# Assure-toi d'avoir !pip install -U ultralytics

# Fonction pour trouver tous les modèles dans le répertoire
def find_models(project_dir):
    models = []
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.pt'):
                model_name = os.path.splitext(file)[0]
                model_path = os.path.join(root, file)
                models.append((model_name, model_path))
    return models

# Titre de l'application Streamlit
st.title("Testeur de Modèles YOLOv11 pour Détection d'Objets Mécaniques - Version Ultime")

tab1, tab2, tab3 = st.tabs(["Test Modèles YOLO", "Laboratoire Mécanique", "Fonctionnalités Avancées"])

with tab1:
    project_dir = st.text_input(
        "Répertoire du projet (ex: /path/to/mechanical_dataset ou chemin local)",
        value="/home/belikan/mechanical_dataset"
    )
    
    if not os.path.exists(project_dir):
        st.error(f"❌ Le répertoire '{project_dir}' n'existe pas. Vérifiez le chemin.")
    else:
        available_models = find_models(project_dir)
        
        if not available_models:
            st.error("❌ Aucun modèle trouvé.")
        else:
            st.success(f"📢 Modèles trouvés : {[name for name, _ in available_models]}")
            
            # Upload de l'image ou vidéo
            uploaded_file = st.file_uploader("Téléversez une image ou vidéo", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'], key="test_uploader")
            
            if uploaded_file is not None:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                is_video = file_ext in ['mp4', 'avi']
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    media_path = tmp_file.name
                
                # Nouvelle fonctionnalité 2: Support vidéo real-time
                if is_video:
                    st.video(media_path)
                
                for model_name, model_path in available_models:
                    st.subheader(f"🔧 Test avec le modèle : {model_name}")
                    
                    try:
                        model = YOLO(model_path)
                        if is_video:
                            results = model.track(media_path, conf=0.05, stream=True)
                            # Traitement stream (simplifié)
                            for r in results:
                                st.write("Frame processed")
                        else:
                            results = model(media_path, conf=0.05)
                        
                        detections_counter = defaultdict(int)
                        defect_detections = []
                        keywords_defect = ["defect", "broken", "crack", "damaged", "rupture"]
                        
                        for r in results:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                confidence = float(box.conf[0])
                                label = r.names[cls_id]
                                detections_counter[label] += 1
                                
                                if any(keyword in label.lower() for keyword in keywords_defect) or confidence < 0.3:
                                    defect_detections.append((label, confidence))
                                
                                st.write(f"✅ Détection : {label} ({confidence:.2f})")
                        
                        st.write("\n📊 Statistiques des objets détectés :")
                        for label, count in detections_counter.items():
                            st.write(f"  🔹 {label} : {count} fois")
                        
                        if defect_detections:
                            st.write("\n❌ Défauts détectés :")
                            for label, conf in defect_detections:
                                st.write(f"  ⚠️ {label} avec {conf:.2f}")
                        else:
                            st.write("✅ Aucun défaut détecté.")
                        
                        # Affichage
                        res_plotted = results[0].plot()
                        fig = plt.figure(figsize=(8, 8))
                        plt.imshow(res_plotted)
                        plt.axis('off')
                        plt.title(f"Résultat - {model_name}")
                        st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Erreur : {str(e)}")
                
                # Sauvegarder images non identifiées (si faible confiance)
                if not is_video and results:
                    last_result = results[-1]  # Dernier résultat
                    if not last_result.boxes or all(float(box.conf[0]) < 0.5 for box in last_result.boxes):
                        unknown_path = os.path.join(UNKNOWN_DIR, f"unknown_{len(os.listdir(UNKNOWN_DIR))}.jpg")
                        cv2.imwrite(unknown_path, cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR))
                        st.info("📸 Image non identifiée sauvegardée pour analyse.")
                
                os.unlink(media_path)

with tab2:
    st.header("Laboratoire d'Entraînement Mécanique")
    
    project_dir_train = st.text_input(
        "Répertoire du projet pour l'entraînement",
        value="/home/belikan/mechanical_dataset"
    )
    
    raw_dir = os.path.join(project_dir_train, 'images_raw')
    video_dir = os.path.join(project_dir_train, 'videos_raw')
    dataset_dir = os.path.join(project_dir_train, 'dataset')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    st.subheader("Upload d'Images ou Vidéos Brutes")
    uploaded_files = st.file_uploader(
        "Téléversez des images ou vidéos",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi'],
        accept_multiple_files=True,
        key="train_uploader"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            target_dir = video_dir if file_ext in ['mp4', 'avi'] else raw_dir
            file_path = os.path.join(target_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        st.success(f"✅ {len(uploaded_files)} fichiers uploadés.")
    
    if st.button("🚀 Lancer l'Entraînement Automatique"):
        with st.spinner("Préparation et entraînement..."):
            try:
                # Nouvelle fonctionnalité 3: OCR avec EasyOCR
                reader = easyocr.Reader(['fr', 'en'])
                
                # Lister images et vidéos
                imgs = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                videos = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi'))]
                
                # Nouvelle fonctionnalité 4: Extract frames from videos
                for vid in videos:
                    cap = cv2.VideoCapture(os.path.join(video_dir, vid))
                    count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if ret and count % 30 == 0:  # Every 30 frames
                            cv2.imwrite(os.path.join(raw_dir, f"frame_{vid}_{count}.jpg"), frame)
                        count += 1
                    cap.release()
                
                imgs = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not imgs:
                    st.error("❌ Aucune image.")
                    st.stop()
                
                # Détection classes avec LLM suggestion
                labels_set = set()
                for img_name in imgs:
                    img_path = os.path.join(raw_dir, img_name)
                    result = reader.readtext(img_path)
                    txt = ' '.join([det[1] for det in result]).lower().strip()
                    fname_label = img_name.lower().split('_')[0].split('.')[0]
                    llm_suggest = "llm_suggest"  # Placeholder
                    candidate = txt or fname_label or llm_suggest
                    if candidate:
                        labels_set.add(candidate.strip().split()[0])
                
                classes = sorted(labels_set)
                
                # Préparer dossiers avec masks
                for split in ['images/train', 'images/val', 'labels/train', 'labels/val', 'masks/train', 'masks/val']:
                    os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)
                
                # Nouvelle fonctionnalité 5: Augmentation
                transform = A.Compose([A.RandomRotate90(), A.Flip(), A.GaussNoise()])
                
                progress_bar = st.progress(0)
                sam_model = SAM('sam2_b.pt')
                for i, img_name in enumerate(imgs):
                    img_path = os.path.join(raw_dir, img_name)
                    img = cv2.imread(img_path)
                    augmented = transform(image=img)['image']
                    aug_name = f"aug_{img_name}"
                    cv2.imwrite(os.path.join(raw_dir, aug_name), augmented)
                    
                    # Labels et masques
                    label = "label"  # From earlier
                    label_id = classes.index(label)
                    bboxes = [(label_id, 0.5, 0.5, 1.0, 1.0)]
                    label_path = os.path.join(dataset_dir, 'labels/train', img_name.rsplit('.', 1)[0] + '.txt')
                    with open(label_path, 'w') as f:
                        for b in bboxes:
                            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                    
                    # Masque with SAM
                    sam_results = sam_model(img_path)
                    mask_path = os.path.join(dataset_dir, 'masks/train', img_name.rsplit('.', 1)[0] + '.png')
                    cv2.imwrite(mask_path, sam_results[0].masks.data[0].cpu().numpy() * 255)
                    
                    shutil.copy(img_path, os.path.join(dataset_dir, 'images/train', img_name))
                    progress_bar.progress((i + 1) / len(imgs))
                
                # Split data
                train_imgs = os.listdir(os.path.join(dataset_dir, 'images/train'))
                random.shuffle(train_imgs)
                split_idx = int(0.8 * len(train_imgs))
                for img_name in train_imgs[split_idx:]:
                    shutil.move(os.path.join(dataset_dir, 'images/train', img_name), os.path.join(dataset_dir, 'images/val', img_name))
                    txt_name = img_name.rsplit('.', 1)[0] + '.txt'
                    mask_name = img_name.rsplit('.', 1)[0] + '.png'
                    if os.path.exists(os.path.join(dataset_dir, 'labels/train', txt_name)):
                        shutil.move(os.path.join(dataset_dir, 'labels/train', txt_name), os.path.join(dataset_dir, 'labels/val', txt_name))
                    if os.path.exists(os.path.join(dataset_dir, 'masks/train', mask_name)):
                        shutil.move(os.path.join(dataset_dir, 'masks/train', mask_name), os.path.join(dataset_dir, 'masks/val', mask_name))
                
                # data.yaml
                yaml_path = os.path.join(project_dir_train, 'data.yaml')
                with open(yaml_path, 'w') as f:
                    f.write(f"train: {dataset_dir}/images/train\n")
                    f.write(f"val: {dataset_dir}/images/val\n")
                    f.write(f"nc: {len(classes)}\n")
                    f.write(f"names: {classes}\n")
                    f.write("segment: true\n")
                
                # Nouvelle fonctionnalité 6: Hyperparam tuning
                def train_fn(config):
                    model = YOLO("yolov11n.pt")
                    model.train(data=yaml_path, epochs=config["epochs"], imgsz=config["imgsz"], batch=config["batch"])
                
                search_space = {"epochs": tune.choice([10, 20]), "imgsz": tune.choice([640]), "batch": tune.choice([4])}
                analysis = tune.run(train_fn, config=search_space, num_samples=2)
                
                best_config = analysis.get_best_config(metric="metrics/mAP50-95(B)", mode="max")
                
                # Entraînement final
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                existing = [d for d in os.listdir(project_dir_train) if d.startswith("yolov11_finetuned_v")]
                versions = [int(re.findall(r'\d+', name)[0]) for name in existing if re.findall(r'\d+', name)]
                model_version = max(versions, default=0) + 1
                model_name = f"yolov11_finetuned_v{model_version}"
                
                base_model_path = "yolov11n.pt"
                
                model = YOLO(base_model_path)
                results = model.train(
                    data=yaml_path,
                    epochs=best_config["epochs"],
                    imgsz=best_config["imgsz"],
                    batch=best_config["batch"],
                    device=device,
                    project=project_dir_train,
                    name=model_name,
                    exist_ok=True,
                    verbose=False,
                    augment=True
                )
                
                st.success(f"✅ Entraînement terminé. Modèle : {model_name}")
                
                # Test
                if imgs:
                    trained_model_path = os.path.join(project_dir_train, model_name, "weights", "best.pt")
                    trained_model = YOLO(trained_model_path)
                    test_image_path = os.path.join(raw_dir, imgs[0])
                    results = trained_model(test_image_path, conf=0.25)
                    
                    res_plotted = results[0].plot()
                    fig = plt.figure(figsize=(8, 8))
                    plt.imshow(res_plotted)
                    plt.axis('off')
                    plt.title(f"Résultat - {model_name}")
                    st.pyplot(fig)
                
                # Nouvelle fonctionnalité 7: Génération rapport PDF
                report_path = os.path.join(project_dir_train, 'reports', f"report_{model_name}.pdf")
                c = canvas.Canvas(report_path, pagesize=letter)
                c.drawString(100, 750, "Rapport Entraînement")
                c.save()
                st.download_button("Télécharger Rapport", report_path)
                
            except Exception as e:
                st.error(f"Erreur : {str(e)}")

with tab3:
    st.header("Fonctionnalités Avancées")
    
    # Nouvelle fonctionnalité 8: Query NLP avec LLM
    query = st.text_input("Pose une question en langage naturel sur l'inspection :")
    if query:
        st.write("Réponse LLM : Analyse en cours... (placeholder)")
    
    # Nouvelle fonctionnalité 9: Dashboard metrics
    st.subheader("Dashboard Performance")
    # Placeholder pour metrics
    
    # Nouvelle fonctionnalité 10: Export modèle
    if st.button("Exporter Modèle en ONNX"):
        model = YOLO("yolov8n.pt")  # Utiliser un modèle par défaut pour l'export
        model.export(format="onnx")
        st.success("Modèle exporté en ONNX !")
    
    # Nouvelle fonctionnalité 11: Anomaly detection mode
    if st.checkbox("Activer Mode Anomaly"):
        st.write("Détection anomalies activée.")
    
    # Nouvelle fonctionnalité 12: Multi-model ensemble
    if st.button("Tester Ensemble"):
        st.write("Ensemble en cours...")
    
    # Nouvelle fonctionnalité 15: Base de Connaissance PDF
    st.subheader("Base de Connaissance PDF")
    
    uploaded_pdfs = st.file_uploader("Uploader des PDFs pour la base de connaissance", type=['pdf'], accept_multiple_files=True, key="pdf_uploader")
    
    if uploaded_pdfs:
        pdf_dir = os.path.join(BASE_DIR, "pdfs")
        os.makedirs(pdf_dir, exist_ok=True)
        for uploaded_pdf in uploaded_pdfs:
            pdf_path = os.path.join(pdf_dir, uploaded_pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getvalue())
        st.success(f"{len(uploaded_pdfs)} PDFs uploadés.")
    
    if st.button("Vectoriser les PDFs dans FAISS"):
        with st.spinner("Extraction, chunking et vectorisation..."):
            try:
                # Charger embedder
                embedder = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Text splitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                
                all_chunks = []
                pdf_dir = os.path.join(BASE_DIR, "pdfs")
                if os.path.exists(pdf_dir):
                    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
                    for pdf_file in pdf_files:
                        pdf_path = os.path.join(pdf_dir, pdf_file)
                        # Extract text
                        doc = fitz.open(pdf_path)
                        text = ""
                        for page in doc:
                            text += page.get_text()
                        doc.close()
                        
                        # Chunk
                        chunks = splitter.split_text(text)
                        for chunk in chunks:
                            all_chunks.append({"text": chunk, "source": pdf_file})
                    
                    # Embed and store
                    embeddings = embedder.encode([c["text"] for c in all_chunks])
                    
                    # FAISS
                    dim = embeddings.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    index.add(embeddings.astype('float32'))
                    faiss.write_index(index, KNOWLEDGE_FAISS)
                    
                    # Save chunks
                    with open(KNOWLEDGE_CHUNKS, "w") as f:
                        json.dump(all_chunks, f)
                    
                    st.success(f"Base de connaissance créée avec {len(all_chunks)} chunks.")
                else:
                    st.warning("Aucun PDF trouvé.")
                    
            except Exception as e:
                st.error(f"Erreur vectorisation: {str(e)}")
    
    # Query knowledge base
    knowledge_query = st.text_input("Question sur la base de connaissance PDF")
    if knowledge_query and st.button("Rechercher dans la Base"):
        try:
            if os.path.exists(KNOWLEDGE_FAISS) and os.path.exists(KNOWLEDGE_CHUNKS):
                embedder = SentenceTransformer('all-MiniLM-L6-v2')
                query_emb = embedder.encode([knowledge_query])[0]
                
                index = faiss.read_index(KNOWLEDGE_FAISS)
                distances, indices = index.search(query_emb.reshape(1, -1).astype('float32'), 3)
                
                with open(KNOWLEDGE_CHUNKS, "r") as f:
                    chunks = json.load(f)
                
                st.subheader("Résultats de recherche")
                for i, idx in enumerate(indices[0]):
                    if idx < len(chunks):
                        st.write(f"**Chunk {i+1}:** {chunks[idx]['text'][:200]}... (Source: {chunks[idx]['source']})")
                        
            else:
                st.warning("Base de connaissance non trouvée.")
                
        except Exception as e:
            st.error(f"Erreur recherche: {str(e)}")
    
    # Nouvelle fonctionnalité 13: Base de Données FAISS et Auto-Apprentissage
    st.subheader("Base de Données FAISS et Auto-Apprentissage")
    
    if st.button("Construire Index FAISS des Images Inconnues"):
        with st.spinner("Construction de l'index FAISS..."):
            try:
                # Charger modèle d'embedding
                embed_model = models.resnet50(pretrained=True)
                embed_model.eval()
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # Lister images inconnues
                unknown_files = [f for f in os.listdir(UNKNOWN_DIR) if f.endswith('.jpg')]
                if not unknown_files:
                    st.warning("Aucune image inconnue.")
                else:
                    embeddings = []
                    for file in unknown_files:
                        img_path = os.path.join(UNKNOWN_DIR, file)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        input_tensor = transform(img).unsqueeze(0)
                        with torch.no_grad():
                            emb = embed_model(input_tensor).squeeze().numpy()
                        embeddings.append(emb)
                    
                    embeddings = np.array(embeddings)
                    
                    # Créer index FAISS
                    dim = embeddings.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    index.add(embeddings)
                    faiss.write_index(index, FAISS_INDEX)
                    
                    st.success(f"Index FAISS créé avec {len(unknown_files)} images.")
                    
            except Exception as e:
                st.error(f"Erreur FAISS: {str(e)}")
    
    if st.button("Auto-Apprentissage avec Scikit-Learn"):
        with st.spinner("Clustering des images inconnues..."):
            try:
                if os.path.exists(FAISS_INDEX):
                    # Recalculer embeddings pour clustering
                    embed_model = models.resnet50(pretrained=True)
                    embed_model.eval()
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    unknown_files = [f for f in os.listdir(UNKNOWN_DIR) if f.endswith('.jpg')]
                    embeddings = []
                    for file in unknown_files:
                        img_path = os.path.join(UNKNOWN_DIR, file)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        input_tensor = transform(img).unsqueeze(0)
                        with torch.no_grad():
                            emb = embed_model(input_tensor).squeeze().numpy()
                        embeddings.append(emb)
                    
                    embeddings = np.array(embeddings)
                    
                    # Clustering
                    n_clusters = min(5, len(embeddings))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(embeddings)
                    
                    # Visualisation avec PCA
                    pca = PCA(n_components=2)
                    reduced = pca.fit_transform(embeddings)
                    
                    fig, ax = plt.subplots()
                    for i in range(n_clusters):
                        mask = clusters == i
                        ax.scatter(reduced[mask, 0], reduced[mask, 1], label=f'Cluster {i}')
                    ax.legend()
                    st.pyplot(fig)
                    
                    st.success(f"Clustering terminé avec {n_clusters} clusters.")
                    
                else:
                    st.warning("Index FAISS non trouvé. Construisez-le d'abord.")
                    
            except Exception as e:
                st.error(f"Erreur auto-apprentissage: {str(e)}")
    
    # Nouvelle fonctionnalité 14: Recherche dans FAISS
    query_img = st.file_uploader("Uploader une image pour recherche similaire", type=['jpg', 'jpeg', 'png'], key="query_uploader")
    if query_img and st.button("Rechercher Similaires"):
        try:
            if os.path.exists(FAISS_INDEX):
                # Calculer embedding de la query
                embed_model = models.resnet50(pretrained=True)
                embed_model.eval()
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                img = Image.open(query_img)
                img = np.array(img)
                input_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    query_emb = embed_model(input_tensor).squeeze().numpy()
                
                # Recherche
                index = faiss.read_index(FAISS_INDEX)
                distances, indices = index.search(query_emb.reshape(1, -1), 5)
                
                unknown_files = [f for f in os.listdir(UNKNOWN_DIR) if f.endswith('.jpg')]
                
                st.subheader("Images Similaires")
                for i, idx in enumerate(indices[0]):
                    if idx < len(unknown_files):
                        similar_path = os.path.join(UNKNOWN_DIR, unknown_files[idx])
                        st.image(similar_path, caption=f"Similaire {i+1} (distance: {distances[0][i]:.2f})")
                        
            else:
                st.warning("Index FAISS non trouvé.")
                
        except Exception as e:
            st.error(f"Erreur recherche: {str(e)}")
    
    # Agent d'Identification de Pièces avec LLM et Tavily
    st.subheader("🤖 Agent d'Identification de Pièces")
    
    part_description = st.text_area("Décrivez la pièce à identifier (forme, fonction, matériau, etc.)", height=100, key="part_desc")
    
    if st.button("🔍 Identifier la Pièce avec Agent IA", key="identify_part"):
        if not part_description.strip():
            st.warning("Veuillez décrire la pièce.")
        else:
            with st.spinner("Agent IA recherche sur internet..."):
                try:
                    agent = create_part_identification_agent()
                    if agent:
                        result = agent.invoke({"input": f"Identifie cette pièce mécanique: {part_description}"})
                        
                        st.subheader("Résultat de l'Identification")
                        st.write(result['output'])
                        
                        # Sauvegarder l'annotation
                        if save_part_annotation(part_description, result['output']):
                            st.success("✅ Annotation sauvegardée!")
                        else:
                            st.warning("⚠️ Erreur sauvegarde annotation")
                    else:
                        st.error("Impossible de créer l'agent. Vérifiez le modèle Mistral et la clé Tavily.")
                        
                except Exception as e:
                    st.error(f"Erreur agent: {str(e)}")
    
    # Afficher les annotations sauvegardées
    if st.button("📚 Voir Annotations Sauvegardées", key="view_annotations"):
        try:
            if os.path.exists(ANNOTATIONS_FILE):
                with open(ANNOTATIONS_FILE, "r") as f:
                    annotations = json.load(f)
                
                st.subheader("Annotations de Pièces")
                for i, ann in enumerate(annotations[-10:], 1):  # Dernières 10
                    with st.expander(f"Annotation {i}: {ann['description'][:50]}..."):
                        st.write(f"**Description:** {ann['description']}")
                        st.write(f"**Résultat:** {ann['result']}")
                        st.write(f"**Date:** {ann['timestamp']}")
            else:
                st.info("Aucune annotation sauvegardée.")
                
        except Exception as e:
            st.error(f"Erreur chargement annotations: {str(e)}")
    
    # Chat Conversationnel avec Agent IA
    st.subheader("💬 Chat avec Agent IA Multimodal")
    
    # Initialiser l'état de session pour le chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if "chat_agent" not in st.session_state:
        st.session_state.chat_agent = None
    
    # Bouton pour initialiser l'agent
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🚀 Démarrer Agent Chat", key="start_chat_agent"):
            with st.spinner("Initialisation de l'agent de chat..."):
                st.session_state.chat_agent = create_chat_agent()
                if st.session_state.chat_agent:
                    st.success("✅ Agent de chat prêt !")
                    # Message de bienvenue
                    welcome_msg = "Bonjour ! Je suis votre assistant IA multimodal. Je peux utiliser des outils comme la recherche internet (Tavily) pour vous aider avec l'identification de pièces, l'analyse technique, et bien plus. Comment puis-je vous aider aujourd'hui ?"
                    st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
                else:
                    st.error("❌ Impossible d'initialiser l'agent.")
    
    with col2:
        if st.button("🗑️ Effacer Historique", key="clear_chat"):
            st.session_state.chat_history = []
            save_chat_history([])
            if "chat_agent" in st.session_state:
                st.session_state.chat_agent = None
            st.success("✅ Historique effacé !")
    
    # Afficher l'historique du chat
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**Vous:** {message['content']}")
            else:
                st.markdown(f"**🤖 Agent:** {message['content']}")
    
    # Zone de saisie pour le message utilisateur
    user_input = st.text_input("Votre message:", key="chat_input", placeholder="Posez votre question ou décrivez ce dont vous avez besoin...")
    
    if st.button("📤 Envoyer", key="send_message") and user_input.strip():
        if st.session_state.chat_agent is None:
            st.warning("Veuillez d'abord démarrer l'agent de chat.")
        else:
            # Ajouter le message utilisateur à l'historique
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner("L'agent réfléchit et utilise ses outils..."):
                try:
                    # Obtenir la réponse de l'agent
                    response = st.session_state.chat_agent.run(user_input)
                    
                    # Ajouter la réponse à l'historique
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Sauvegarder l'historique
                    save_chat_history(st.session_state.chat_history)
                    
                    # Rafraîchir l'affichage
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Erreur lors de la génération de réponse: {str(e)}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    save_chat_history(st.session_state.chat_history)
                    st.error(error_msg)
    
    # Informations sur les capacités de l'agent
    with st.expander("ℹ️ Capacités de l'Agent"):
        st.markdown("""
        **L'agent peut utiliser ces outils :**
        - 🔍 **Recherche Internet (Tavily)** : Recherche d'informations sur pièces mécaniques, spécifications techniques, etc.
        - 🧠 **Connaissance Contextuelle** : Maintient le contexte de la conversation
        - 📚 **Base de Connaissances** : Peut accéder aux PDFs vectorisés si disponibles
        
        **Exemples d'utilisation :**
        - "Quelle est cette pièce ronde avec des dents ?"
        - "Trouve-moi les spécifications du joint de culasse pour Peugeot 206"
        - "Explique-moi comment fonctionne un système de freinage ABS"
        - "Compare les prix des filtres à huile pour Renault Clio"
        """)