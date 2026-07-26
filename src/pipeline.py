from src.audio_preprocessing import preprocess_audio
from src.asr_model import transcribe
from src.sentiment_model import analyze_sentiment


def run_pipeline(file_path: str) -> dict:
    """
    Exécute le pipeline complet : audio -> texte -> sentiment.
    Lève une ValueError avec un message clair en cas de problème
    (fichier invalide, audio vide/silencieux, transcription vide, etc.).
    """
    # Étape 1 : prétraitement audio (peut lever ValueError)
    audio_array, sample_rate = preprocess_audio(file_path)

    # Étape 2 : transcription
    transcription = transcribe(audio_array, sample_rate)

    if not transcription or not transcription.strip():
        raise ValueError(
            "La transcription est vide : impossible d'analyser le sentiment."
        )

    # Étape 3 : analyse de sentiment
    sentiment_result = analyze_sentiment(transcription)

    # Étape 4 : assemblage du résultat final (format attendu par l'API)
    return {
        "transcription": transcription,
        "sentiment": sentiment_result["label"],
        "confidence": sentiment_result["score"],
    }