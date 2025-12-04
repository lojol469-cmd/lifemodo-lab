import os
import json
import glob
from caption_auto import generate_caption
import streamlit as st

AUDIO_DIR = "temp_audio_validated"
OUTPUT_JSON = "dataset_musicgen.json"

def create_musicgen_dataset(audio_directory=AUDIO_DIR, output_file=OUTPUT_JSON, progress_callback=None):
    """
    Crée un dataset MusicGen à partir des fichiers audio du dossier spécifié.
    Génère automatiquement des captions pour chaque fichier audio.
    """
    dataset = []

    # Trouver tous les fichiers audio
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.aac']:
        audio_files.extend(glob.glob(os.path.join(audio_directory, ext)))

    if not audio_files:
        print(f"❌ Aucun fichier audio trouvé dans {audio_directory}")
        return None

    print(f"📊 Trouvé {len(audio_files)} fichiers audio")
    print("🎵 Génération des captions musicales...")

    for i, audio_file in enumerate(audio_files):
        try:
            print(f"  [{i+1}/{len(audio_files)}] Traitement: {os.path.basename(audio_file)}")

            # Générer la caption
            caption = generate_caption(audio_file)

            # Ajouter au dataset
            dataset.append({
                "audio": audio_file,
                "text": caption,
                "file": os.path.basename(audio_file)
            })

            # Callback de progression si fourni
            if progress_callback:
                progress_callback((i+1) / len(audio_files), f"Génération caption pour {os.path.basename(audio_file)}")

            print(f"    ✅ Caption générée: {caption[:100]}...")

        except Exception as e:
            print(f"    ❌ Erreur avec {audio_file}: {e}")
            continue

    # Sauvegarder le dataset
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"✅ Dataset MusicGen créé: {output_file}")
    print(f"📊 {len(dataset)} exemples dans le dataset")

    return dataset

if __name__ == "__main__":
    # Création du dataset en mode standalone
    print("🎵 Création du dataset MusicGen pour Tcham AI...")
    dataset = create_musicgen_dataset()

    if dataset:
        print("\n📋 Aperçu du dataset:")
        for i, item in enumerate(dataset[:3]):  # Montrer les 3 premiers
            print(f"{i+1}. {item['file']}")
            print(f"   Caption: {item['text'][:150]}...")
            print()

        print("🎉 Dataset prêt pour l'entraînement MusicGen!")
    else:
        print("❌ Échec de la création du dataset")