import os
import json
import torch
from datasets import Dataset, Audio
from transformers import (
    AutoProcessor,
    MusicgenForConditionalGeneration,
    TrainingArguments,
    Trainer
)
import numpy as np

MODEL_NAME = "facebook/musicgen-small"
DATASET_JSON = "dataset_musicgen.json"
OUTPUT_DIR = "musicgen-tcham-v1"

def load_musicgen_dataset(json_file):
    """Charge le dataset MusicGen depuis le fichier JSON"""
    print(f"📂 Chargement du dataset depuis {json_file}...")

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ {len(data)} exemples chargés")

    # Convertir en Dataset HF
    ds = Dataset.from_list(data)

    return ds

def setup_lora_model(model_name):
    """Configure le modèle MusicGen avec LoRA"""
    print(f"🤖 Chargement du modèle {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    # Configuration LoRA pour MusicGen
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=16,  # Rang LoRA
        lora_alpha=32,
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],  # Modules d'attention
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    # Appliquer LoRA
    model = get_peft_model(model, lora_config)

    print(f"✅ LoRA configuré - Paramètres entraînables: {model.num_parameters(only_trainable=True)}")

    return processor, model

def preprocess_function(batch, processor):
    """Préprocessing des données pour MusicGen"""
    audio_paths = batch["audio"]
    texts = batch["text"]

    # Charger les fichiers audio
    audio_arrays = []
    for audio_path in audio_paths:
        try:
            # Utiliser librosa pour charger l'audio
            import librosa
            y, sr = librosa.load(audio_path, sr=16000)

            # Tronquer à 30 secondes max pour MusicGen
            max_length = 30 * 16000
            if len(y) > max_length:
                y = y[:max_length]

            audio_arrays.append(y)
        except Exception as e:
            print(f"Erreur chargement {audio_path}: {e}")
            # Audio vide en cas d'erreur
            audio_arrays.append(np.zeros(16000, dtype=np.float32))

    # Tokeniser les textes
    text_inputs = processor(
        text=texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Préparer les inputs audio
    # MusicGen attend des tensors audio normalisés
    audio_tensors = []
    for audio in audio_arrays:
        # Normaliser l'audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Convertir en tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        audio_tensors.append(audio_tensor)

    # Padding des séquences audio
    max_audio_len = max(len(a) for a in audio_tensors)
    padded_audios = []
    attention_masks = []

    for audio in audio_tensors:
        # Padding
        padding_length = max_audio_len - len(audio)
        if padding_length > 0:
            padded_audio = torch.cat([audio, torch.zeros(padding_length)])
        else:
            padded_audio = audio

        padded_audios.append(padded_audio)

        # Attention mask
        mask = torch.ones(len(audio))
        if padding_length > 0:
            mask = torch.cat([mask, torch.zeros(padding_length)])
        attention_masks.append(mask)

    return {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "audio_values": torch.stack(padded_audios),
        "audio_attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(padded_audios)  # Pour l'entraînement génératif
    }

def train_musicgen_lora(dataset_json=DATASET_JSON, output_dir=OUTPUT_DIR):
    """Entraînement complet du modèle MusicGen avec LoRA"""

    # Vérifier si CUDA est disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Entraînement sur: {device}")

    # Charger le dataset
    dataset = load_musicgen_dataset(dataset_json)
    if dataset is None:
        return False

    # Diviser en train/validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Configurer le modèle et le processor
    processor, model = setup_lora_model(MODEL_NAME)

    # Déplacer le modèle sur le device
    model = model.to(device)

    # Fonction de preprocessing
    def preprocess_batch(batch):
        return preprocess_function(batch, processor)

    # Appliquer le preprocessing
    print("🔄 Préprocessing des données...")
    train_dataset = train_dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=4,
        remove_columns=train_dataset.column_names
    )

    eval_dataset = eval_dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=4,
        remove_columns=eval_dataset.column_names
    )

    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Petit batch size pour MusicGen
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Accumulation pour simuler un batch plus grand
        learning_rate=2e-4,
        num_train_epochs=10,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True if device == "cuda" else False,  # Mixed precision si GPU
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # Créer le trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
    )

    # Lancer l'entraînement
    print("🚀 Début de l'entraînement MusicGen LoRA...")
    trainer.train()

    # Sauvegarder le modèle final
    print(f"💾 Sauvegarde du modèle dans {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    print("🎉 Entraînement terminé avec succès!")
    print(f"📁 Modèle sauvegardé dans: {output_dir}")

    return True

if __name__ == "__main__":
    print("🎵 Entraînement MusicGen-Tcham-v1 avec LoRA")
    print("=" * 50)

    success = train_musicgen_lora()

    if success:
        print("\n🎊 Félicitations! MusicGen-Tcham-v1 est prêt!")
        print("📋 Pour générer de la musique, utilisez inference_musicgen.py")
    else:
        print("\n❌ Échec de l'entraînement")