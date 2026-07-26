import librosa
import numpy as np


def preprocess_audio(file_path, target_sample_rate=16000, max_duration_sec=300):
    """
    Charge un fichier audio, le convertit en mono, le rééchantillonne à 16kHz,
    normalise l'amplitude, et vérifie qu'il est valide.
    """
    # 1. Chargement + rééchantillonnage + conversion mono en une seule étape
    #    librosa gère nativement .wav, .mp3, .flac, etc.
    try:
        audio_array, _ = librosa.load(file_path, sr=target_sample_rate, mono=True)
    except Exception as e:
        raise ValueError(f"Impossible de charger le fichier audio : {e}")

    # 2. Vérification que le fichier n'est pas vide
    if audio_array.size == 0:
        raise ValueError("Le fichier audio est vide.")

    # 3. Vérification de la durée
    duration_sec = len(audio_array) / target_sample_rate
    if duration_sec > max_duration_sec:
        raise ValueError(f"Durée trop longue : {duration_sec:.1f}s (max {max_duration_sec}s)")

    # 4. Détection de silence total
    if np.max(np.abs(audio_array)) < 1e-4:
        raise ValueError("Le fichier audio semble silencieux.")

    # 5. Normalisation (zero-mean, unit-variance)
    audio_array = (audio_array - np.mean(audio_array)) / (np.std(audio_array) + 1e-8)

    return audio_array, target_sample_rate