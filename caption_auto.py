import librosa
import numpy as np
from transformers import pipeline

captioner = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

def generate_caption(audio_path):
    """
    Génère automatiquement une caption musicale détaillée pour un fichier audio.
    Analyse les caractéristiques audio et crée un prompt descriptif.
    """
    try:
        # Charger l'audio
        y, sr = librosa.load(audio_path, sr=16000)

        # Analyser les caractéristiques musicales
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        rms = np.mean(librosa.feature.rms(y=y))

        # Construire le prompt d'analyse
        prompt = f"""
        Analyze this music piece and create a detailed caption for music generation:

        Audio features:
        - Tempo: {tempo:.1f} BPM
        - Spectral centroid: {spectral_centroid:.1f}
        - Spectral rolloff: {rolloff:.1f}
        - RMS energy: {rms:.3f}
        - Chroma features: {', '.join([f'{i}:{v:.2f}' for i, v in enumerate(chroma)])}

        The style is African Tcham dance music with rhythmic percussion, energetic grooves, and traditional African instruments.

        Create a detailed musical description that could be used as a prompt for music generation. Include:
        - Musical style and cultural elements
        - Instrumentation (drums, percussion, bass, etc.)
        - Rhythm and tempo description
        - Mood and energy level
        - Specific Tcham dance characteristics

        Caption:"""

        # Générer la caption avec le modèle
        out = captioner(prompt, max_new_tokens=120, temperature=0.7, do_sample=True)[0]["generated_text"]

        # Nettoyer et retourner la caption
        caption = out.split("Caption:")[-1].strip() if "Caption:" in out else out.strip()
        caption = caption.replace("\n", " ").strip()

        # Assurer une longueur minimale
        if len(caption) < 20:
            caption = f"African Tcham dance beat with energetic percussion, tempo {tempo:.0f} BPM, featuring traditional African rhythms and instruments."

        return caption

    except Exception as e:
        print(f"Erreur lors de la génération de caption pour {audio_path}: {e}")
        return "African Tcham dance music with rhythmic percussion and energetic grooves, traditional African instruments and beats."