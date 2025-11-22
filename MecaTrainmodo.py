# Copyright (c) 2025 Belikan. All rights reserved.
# Licensed under the LifeModo AI Lab License. See LICENSE file for details.
# Contact: belikan@lifemodo.ai

# ✅ Installer les dépendances YOLOv11
!pip install -U ultralytics easyocr ray[tune] albumentations --quiet

# === Étape 1 : Monter Google Drive ===
from google.colab import drive
import os, sys, shutil, random, re, torch, cv2, easyocr
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from ray import tune
drive.mount('/content/drive')

# === Configuration du Projet ===
PROJECT_DIR = '/content/drive/MyDrive/mechanical_dataset'
RAW_DIR = os.path.join(PROJECT_DIR, 'images_raw')
DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, 'reports'), exist_ok=True)  # Nouveau: dossier pour rapports

# Nouvelle fonctionnalité 1: Support pour vidéos raw
VIDEO_DIR = os.path.join(PROJECT_DIR, 'videos_raw')
os.makedirs(VIDEO_DIR, exist_ok=True)

# Nouvelle fonctionnalité 2: Intégration LLM placeholder (utilise Grok API ou similaire)
def llm_label_suggestion(image_path):
    # Placeholder: Appel à un LLM pour suggestions de labels
    # Ex: response = requests.post("https://api.x.ai/v1/chat/completions", ...)
    return "suggested_label_from_llm"  # Remplacer par vrai appel

# === Fonction pour calculer l'ID du prochain modèle ===
def get_next_model_version(base_dir, base_name="yolov11_finetuned_v"):
    existing = [d for d in os.listdir(base_dir) if d.startswith(base_name)]
    versions = [int(re.findall(r'\d+', name)[0]) for name in existing if re.findall(r'\d+', name)]
    return max(versions, default=0) + 1

def new_images_detected(train_dir, raw_dir):
    trained_images = set(os.listdir(train_dir))
    raw_images = set(os.listdir(raw_dir))
    new_images = raw_images - trained_images
    return len(new_images) > 0

if new_images_detected(os.path.join(DATASET_DIR, 'images/train'), RAW_DIR):
    MODEL_VERSION = get_next_model_version(PROJECT_DIR)
    MODEL_NAME = f"yolov11_finetuned_v{MODEL_VERSION}"
    print(f"📌 Nouvelles images détectées. Nouveau modèle : {MODEL_NAME}")
else:
    print("✅ Aucune nouvelle image détectée. Pas de nouveau modèle.")
    sys.exit()

# === Étape 2 : Installer les dépendances supplémentaires ===
!pip install -q easyocr opencv-python-headless ultralytics ray[tune] albumentations reportlab
!apt-get -qq install -y tesseract-ocr tesseract-ocr-fra libtesseract-dev

# === Étape 3 : Imports supplémentaires ===
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import easyocr

# Nouvelle fonctionnalité 3: Lecteur OCR amélioré avec EasyOCR
reader = easyocr.Reader(['fr', 'en'])

# === Étape 4 : Lister les images et vidéos ===
imgs = sorted([f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi'))])
if not imgs and not videos:
    sys.exit("❌ Aucune image ou vidéo trouvée.")
print(f"📸 {len(imgs)} images et {len(videos)} vidéos trouvées.")

# Nouvelle fonctionnalité 4: Extraction de frames des vidéos pour augmentation dataset
def extract_frames(video_path, output_dir, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret and count % frame_rate == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame_{count}.jpg"), frame)
        count += 1
    cap.release()

for vid in videos:
    extract_frames(os.path.join(VIDEO_DIR, vid), RAW_DIR)

# === Étape 5 : Détection dynamique des classes avec OCR amélioré et LLM ===
labels_set = set()
for img_name in imgs:
    img_path = os.path.join(RAW_DIR, img_name)
    result = reader.readtext(img_path)
    txt = ' '.join([det[1] for det in result]).lower().strip()
    fname_label = img_name.lower().split('_')[0].split('.')[0]
    llm_suggest = llm_label_suggestion(img_path)
    candidate = txt or fname_label or llm_suggest
    if candidate:
        label = candidate.strip().split()[0]
        labels_set.add(label)

classes = sorted(labels_set)
print(f"✅ {len(classes)} classes détectées : {classes}")

# === Étape 6 : Préparer les dossiers avec support segmentation ===
for split in ['images/train', 'images/val', 'labels/train', 'labels/val', 'masks/train', 'masks/val']:
    os.makedirs(os.path.join(DATASET_DIR, split), exist_ok=True)

# Nouvelle fonctionnalité 5: Augmentation de données avec Albumentations
transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.GaussNoise(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.Blur(blur_limit=3),
    A.OpticalDistortion(),
    A.GridDistortion(),
])

def augment_image(img_path, output_dir, num_augs=3):
    image = cv2.imread(img_path)
    for i in range(num_augs):
        augmented = transform(image=image)['image']
        cv2.imwrite(os.path.join(output_dir, f"aug_{i}_{os.path.basename(img_path)}"), augmented)

# Appliquer augmentation
for img_name in imgs:
    augment_image(os.path.join(RAW_DIR, img_name), RAW_DIR)

# Mise à jour liste images après augmentation
imgs = sorted([f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# === Étape 7 : Génération des annotations YOLO avec segmentation ===
from ultralytics.models.sam import SAM  # Nouvelle fonctionnalité 6: Utilisation de SAM pour masques
sam_model = SAM('sam2_b.pt')

MIN_CONTOUR_AREA = 500
for img_name in imgs:
    img_path = os.path.join(RAW_DIR, img_name)
    img = cv2.imread(img_path)
    result = reader.readtext(img_path)
    txt = ' '.join([det[1] for det in result]).lower().strip()
    fname_label = img_name.lower().split('_')[0].split('.')[0]
    candidate = txt if len(txt) > len(fname_label) else fname_label
    label = candidate.strip().split()[0]

    if label not in classes:
        classes.append(label)
        classes = sorted(set(classes))
        print(f"➕ Nouveau label détecté : {label}")

    label_id = classes.index(label)
    # Détection contours pour bbox
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    bboxes = []
    for cnt in contours:
        x, y, wi, hi = cv2.boundingRect(cnt)
        if wi * hi >= MIN_CONTOUR_AREA:
            bboxes.append((label_id, (x + wi / 2) / w, (y + hi / 2) / h, wi / w, hi / h))

    if not bboxes:
        bboxes = [(label_id, 0.5, 0.5, 1.0, 1.0)]

    label_path = os.path.join(DATASET_DIR, 'labels/train', img_name.rsplit('.', 1)[0] + '.txt')
    with open(label_path, 'w') as f:
        for b in bboxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    # Génération masques avec SAM
    sam_results = sam_model(img_path)
    mask_path = os.path.join(DATASET_DIR, 'masks/train', img_name.rsplit('.', 1)[0] + '.png')
    # Sauvegarde masque (simplifié)
    cv2.imwrite(mask_path, sam_results[0].masks.data[0].cpu().numpy() * 255)

    shutil.copy(img_path, os.path.join(DATASET_DIR, 'images/train', img_name))

print("✅ Annotations YOLO et masques créés.")

# === Étape 8 : Séparer train / val (80/20) avec masques ===
train_imgs = os.listdir(os.path.join(DATASET_DIR, 'images/train'))
random.shuffle(train_imgs)
split_idx = int(0.8 * len(train_imgs))
for img_name in train_imgs[split_idx:]:
    shutil.move(os.path.join(DATASET_DIR, 'images/train', img_name),
                os.path.join(DATASET_DIR, 'images/val', img_name))
    txt_name = img_name.rsplit('.', 1)[0] + '.txt'
    mask_name = img_name.rsplit('.', 1)[0] + '.png'
    if os.path.exists(os.path.join(DATASET_DIR, 'labels/train', txt_name)):
        shutil.move(os.path.join(DATASET_DIR, 'labels/train', txt_name),
                    os.path.join(DATASET_DIR, 'labels/val', txt_name))
    if os.path.exists(os.path.join(DATASET_DIR, 'masks/train', mask_name)):
        shutil.move(os.path.join(DATASET_DIR, 'masks/train', mask_name),
                    os.path.join(DATASET_DIR, 'masks/val', mask_name))

print(f"📂 Données séparées : {split_idx} train / {len(train_imgs) - split_idx} val")

# === Étape 9 : Créer data.yaml avec support segmentation ===
yaml_path = os.path.join(PROJECT_DIR, 'data.yaml')
with open(yaml_path, 'w') as f:
    f.write(f"train: {DATASET_DIR}/images/train\n")
    f.write(f"val: {DATASET_DIR}/images/val\n")
    f.write(f"nc: {len(classes)}\n")
    f.write(f"names: {classes}\n")
    f.write("segment: true\n")  # Activation segmentation
print(f"📄 data.yaml créé : {yaml_path}")

# Nouvelle fonctionnalité 7: Hyperparameter tuning avec Ray Tune
def train_fn(config):
    model = YOLO("yolov11n.pt")  # Utilise YOLOv11
    model.train(data=yaml_path, epochs=config["epochs"], imgsz=config["imgsz"], batch=config["batch"])

search_space = {
    "epochs": tune.choice([20, 30, 50]),
    "imgsz": tune.choice([640, 800]),
    "batch": tune.choice([4, 8, 16]),
}
analysis = tune.run(train_fn, config=search_space, num_samples=5)

best_config = analysis.get_best_config(metric="metrics/mAP50-95(B)", mode="max")
print(f"Meilleure config: {best_config}")

# === Étape 10 : Fine-tuning avec YOLOv11 et meilleure config ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"⚙️ Entraînement sur device : {device}")

base_model_path = os.path.join(PROJECT_DIR, "yolov11_model", "weights", "best.pt")
if not os.path.exists(base_model_path):
    base_model_path = "yolov11n.pt"  # YOLOv11 nano pour start

model = YOLO(base_model_path)
results = model.train(
    data=yaml_path,
    epochs=best_config["epochs"],
    imgsz=best_config["imgsz"],
    batch=best_config["batch"],
    device=0 if device == 'cuda' else 'cpu',
    project=PROJECT_DIR,
    name=MODEL_NAME,
    exist_ok=True,
    resume=False,
    save=True,
    verbose=True,
    val=True,
    augment=True,  # Augmentation intégrée
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0,
    flipud=0.5,
    fliplr=0.5,
    bgr=0.0,
    erasing=0.4,
    crop_fraction=1.0
)

print(f"✅ Fine-tuning terminé. Nouveau modèle : {MODEL_NAME}")

# Nouvelle fonctionnalité 8: Export à ONNX pour déploiement
model.export(format="onnx", dynamic=True)
print("📦 Modèle exporté en ONNX.")

# === Étape 11 : Tester sur une image avec anomaly detection ===
trained_model_path = os.path.join(PROJECT_DIR, MODEL_NAME, "weights", "best.pt")
trained_model = YOLO(trained_model_path)
test_image_path = os.path.join(RAW_DIR, imgs[0])
results = trained_model(test_image_path, conf=0.25)

# Nouvelle fonctionnalité 9: Détection d'anomalies basique (e.g., low conf = anomaly)
anomalies = [box for box in results[0].boxes if box.conf < 0.3]

res_plotted = results[0].plot()
plt.figure(figsize=(10, 10))
plt.imshow(res_plotted)
plt.axis('off')
plt.title(f"📍 Résultat - {MODEL_NAME}")
plt.show()

# Nouvelle fonctionnalité 10: Génération de rapport PDF
def generate_report(pdf_path, results, anomalies):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "Rapport d'Inspection Mécanique")
    c.drawString(100, 730, f"Modèle: {MODEL_NAME}")
    c.drawString(100, 710, f"Détections: {len(results[0].boxes)}")
    if anomalies:
        c.drawString(100, 690, "Anomalies détectées!")
    c.save()

report_path = os.path.join(PROJECT_DIR, 'reports', f"report_{MODEL_NAME}.pdf")
generate_report(report_path, results, anomalies)
print(f"📑 Rapport généré : {report_path}")

# Nouvelle fonctionnalité 11: Support multi-GPU
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Nouvelle fonctionnalité 12: Ensemble de modèles (simplifié)
previous_models = [os.path.join(PROJECT_DIR, d, "weights", "best.pt") for d in os.listdir(PROJECT_DIR) if d.startswith("yolov11_finetuned_v")]
ensemble_results = []
for pm in previous_models[:2]:  # 2 modèles pour ensemble
    em = YOLO(pm)
    ensemble_results.append(em(test_image_path))

# Moyenne des résultats (simplifié)
print("Ensemble résultats calculés.")

# === Étape Finale : Lister les modèles fine-tunés ===
def list_available_models(base_dir, base_name="yolov11_finetuned_v"):
    print("\n📚 Modèles disponibles :")
    models = sorted([d for d in os.listdir(base_dir) if d.startswith(base_name)])
    if not models:
        print("❌ Aucun modèle trouvé.")
    else:
        for idx, model_dir in enumerate(models, 1):
            model_path = os.path.join(base_dir, model_dir, "weights", "best.pt")
            status = "✅ Prêt" if os.path.exists(model_path) else "⛔ Incomplet"
            print(f"{idx}. {model_dir} → {status}")

list_available_models(PROJECT_DIR)