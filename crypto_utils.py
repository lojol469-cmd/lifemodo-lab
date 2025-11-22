# Copyright (c) 2025 Belikan. All rights reserved.
# Licensed under the LifeModo AI Lab License. See LICENSE file for details.
# Contact: belikan@lifemodo.ai

"""
Utilitaire de cryptage pour protéger les fichiers importants
Utilise le mot de passe "belikan" pour ouvrir les plus importants
"""

from cryptography.fernet import Fernet
import base64
import hashlib
import os

def generate_key(password):
    """Génère une clé à partir du mot de passe 'belikan'"""
    password_bytes = password.encode()
    salt = b'belikan_salt_2025'  # Sel fixe pour cohérence
    key = hashlib.pbkdf2_hmac('sha256', password_bytes, salt, 100000)
    return base64.urlsafe_b64encode(key)

def encrypt_file(file_path, password='belikan'):
    """Crypte un fichier avec le mot de passe"""
    key = generate_key(password)
    fernet = Fernet(key)

    with open(file_path, 'rb') as file:
        original = file.read()

    encrypted = fernet.encrypt(original)

    encrypted_path = file_path + '.encrypted'
    with open(encrypted_path, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)

    print(f"Fichier crypté: {encrypted_path}")
    return encrypted_path

def decrypt_file(encrypted_path, password='belikan'):
    """Décrypte un fichier avec le mot de passe"""
    key = generate_key(password)
    fernet = Fernet(key)

    with open(encrypted_path, 'rb') as encrypted_file:
        encrypted = encrypted_file.read()

    try:
        decrypted = fernet.decrypt(encrypted)

        original_path = encrypted_path.replace('.encrypted', '')
        with open(original_path, 'wb') as decrypted_file:
            decrypted_file.write(decrypted)

        print(f"Fichier décrypté: {original_path}")
        return original_path
    except Exception as e:
        print(f"Erreur de décryptage: {e}")
        return None

# Exemple d'utilisation pour crypter les fichiers importants
if __name__ == "__main__":
    # Crypter les fichiers importants avec "belikan"
    important_files = [
        'models/vision_model/weights/best.pt',
        'models/language_model/pytorch_model.bin',
        'models/audio_model.pt',
        'llms/mistral-7b/model.bin'
    ]

    for file_path in important_files:
        if os.path.exists(file_path):
            encrypt_file(file_path, 'belikan')
            print(f"✅ {file_path} crypté avec succès")
        else:
            print(f"⚠️ {file_path} non trouvé")

    print("\nPour décrypter, utilisez decrypt_file('fichier.encrypted', 'belikan')")