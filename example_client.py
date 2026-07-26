
import requests
import sys

API_URL = "http://127.0.0.1:8000/predict"


def analyze_audio_file(file_path: str) -> dict:
    """Envoie un fichier audio à l'API et retourne le résultat JSON."""
    with open(file_path, "rb") as f:
        response = requests.post(API_URL, files={"file": f})

    if response.status_code != 200:
        raise RuntimeError(f"Erreur API ({response.status_code}) : {response.json()}")

    return response.json()


if __name__ == "__main__":
    # Utilise le fichier passé en argument, sinon un fichier d'exemple par défaut
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/exemple_test.wav"

    print(f"Analyse du fichier : {file_path}\n")
    result = analyze_audio_file(file_path)

    print("Transcription :", result["transcription"])
    print("Sentiment     :", result["sentiment"])
    print("Confiance     :", f"{result['confidence'] * 100:.1f} %")


