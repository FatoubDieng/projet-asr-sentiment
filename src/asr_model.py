from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# Chargement du modèle et du processor (une seule fois, au démarrage de l'app)
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

def transcribe(audio_array, sample_rate=16000):
    """
    Prend un signal audio (numpy array, 16kHz, mono, normalisé)
    et retourne la transcription texte.
    """
    # 1. Préparation de l'entrée avec le processor
    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )

    # 2. Inférence (pas de calcul de gradient nécessaire)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # 3. Décodage : on prend l'indice le plus probable à chaque pas de temps
    predicted_ids = torch.argmax(logits, dim=-1)

    # 4. Conversion des IDs en texte lisible
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription.strip()