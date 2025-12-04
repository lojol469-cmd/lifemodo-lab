import torch
import scipy.io.wavfile as wav
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os

def load_musicgen_model(model_path="musicgen-tcham-v1"):
    """
    Charge le modèle MusicGen entraîné avec LoRA
    """
    print(f"🤖 Chargement du modèle depuis {model_path}...")

    try:
        # Charger le processor
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

        # Charger le modèle entraîné
        model = MusicgenForConditionalGeneration.from_pretrained(model_path)

        print("✅ Modèle MusicGen-Tcham-v1 chargé avec succès!")
        return processor, model

    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("💡 Assurez-vous que le modèle a été entraîné avec train_musicgen_lora.py")
        return None, None

def generate_music(prompt, processor, model, duration=10, guidance_scale=3.0, temperature=1.0):
    """
    Génère de la musique à partir d'un prompt textuel

    Args:
        prompt (str): Description musicale
        processor: Processor MusicGen
        model: Modèle MusicGen
        duration (int): Durée en secondes
        guidance_scale (float): Force du guidance (plus élevé = plus fidèle au prompt)
        temperature (float): Créativité (1.0 = équilibré)

    Returns:
        numpy.ndarray: Audio généré
    """
    print(f"🎵 Génération de musique: '{prompt}'")
    print(f"⏱️ Durée: {duration}s, Guidance: {guidance_scale}, Température: {temperature}")

    # Préparer les inputs
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    )

    # Générer l'audio
    with torch.no_grad():
        # Calculer le nombre de tokens pour la durée souhaitée
        # MusicGen génère environ 50 tokens par seconde
        max_new_tokens = int(duration * 50)

        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            guidance_scale=guidance_scale,
            temperature=temperature,
            do_sample=True,
            num_beams=1  # Sampling au lieu de beam search pour plus de créativité
        )

    # Convertir en numpy array
    audio_array = audio_values[0].cpu().numpy()

    print(f"✅ Musique générée: {len(audio_array)} samples ({len(audio_array)/16000:.1f}s)")

    return audio_array

def save_audio(audio_array, filename, sample_rate=16000):
    """
    Sauvegarde l'audio généré dans un fichier WAV
    """
    # Normaliser l'audio
    if np.max(np.abs(audio_array)) > 0:
        audio_array = audio_array / np.max(np.abs(audio_array))

    # Convertir en int16 pour WAV
    audio_int16 = (audio_array * 32767).astype(np.int16)

    wav.write(filename, sample_rate, audio_int16)
    print(f"💾 Audio sauvegardé: {filename}")

def generate_tcham_music_examples(processor, model, output_dir="generated_music"):
    """
    Génère plusieurs exemples de musique Tcham
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prompts Tcham prédéfinis
    tcham_prompts = [
        "African Tcham dance beat with energetic percussion, traditional African drums, fast tempo around 120 BPM",
        "Tcham music with deep bass, rhythmic hand claps, and traditional African vocal chants",
        "Modern Tcham fusion with electronic elements, keeping the traditional African percussion rhythm",
        "Tcham wedding music with joyful percussion, celebratory atmosphere, multiple drummers",
        "Slow Tcham meditation music with gentle percussion and African wind instruments"
    ]

    generated_files = []

    for i, prompt in enumerate(tcham_prompts):
        print(f"\n🎵 Génération exemple {i+1}/{len(tcham_prompts)}")

        # Générer l'audio
        audio = generate_music(prompt, processor, model, duration=15)

        # Sauvegarder
        filename = os.path.join(output_dir, f"tcham_example_{i+1}.wav")
        save_audio(audio, filename)

        generated_files.append(filename)

    print(f"\n🎉 {len(generated_files)} exemples de musique Tcham générés dans {output_dir}/")

    return generated_files

def interactive_generation(processor, model):
    """
    Mode interactif pour générer de la musique Tcham personnalisée
    """
    print("\n🎵 Mode génération interactive MusicGen-Tcham-v1")
    print("=" * 50)

    while True:
        print("\nOptions:")
        print("1. Générer avec prompt personnalisé")
        print("2. Générer exemples Tcham prédéfinis")
        print("3. Quitter")

        choice = input("\nVotre choix (1-3): ").strip()

        if choice == "1":
            prompt = input("Entrez votre prompt musical: ").strip()
            if prompt:
                duration = int(input("Durée en secondes (5-30): ").strip() or "10")
                duration = max(5, min(30, duration))

                audio = generate_music(prompt, processor, model, duration=duration)

                filename = input("Nom du fichier de sortie (sans extension): ").strip() or "custom_music"
                filename = f"{filename}.wav"

                save_audio(audio, filename)
                print(f"✅ Musique générée et sauvegardée: {filename}")

        elif choice == "2":
            output_dir = input("Dossier de sortie: ").strip() or "tcham_examples"
            generate_tcham_music_examples(processor, model, output_dir)

        elif choice == "3":
            print("👋 Au revoir!")
            break

        else:
            print("❌ Choix invalide")

if __name__ == "__main__":
    print("🎵 MusicGen-Tcham-v1 - Génération de musique")
    print("=" * 50)

    # Charger le modèle
    processor, model = load_musicgen_model()

    if processor is None or model is None:
        print("❌ Impossible de charger le modèle. Assurez-vous qu'il a été entraîné.")
        exit(1)

    # Mode interactif
    interactive_generation(processor, model)