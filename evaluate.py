import re
import jiwer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.audio_preprocessing import preprocess_audio
from src.asr_model import transcribe
from src.sentiment_model import analyze_sentiment

EVAL_DATASET = [
    {
        "file": "data/samples/exemple_test.wav",
        "reference_text": (
            "ce restaurant était vraiment excellent le service était rapide "
            "et l accueil chaleureux je recommande vivement"
        ),
        "expected_sentiment": "positif",
    },
    {
        "file": "data/samples/exemple_negative.wav",
        "reference_text": (
            "cette expérience était vraiment décevante le produit est arrivé "
            "cassé et le service client n a jamais répondu à mes messages"
        ),
        "expected_sentiment": "négatif",
    },
    {
        "file": "data/samples/exemple_neutre.wav",
        "reference_text": (
            "le colis est arrivé mardi à quatorze heures conformément au "
            "délai indiqué lors de la commande"
        ),
        "expected_sentiment": "neutre",
    },
]

 #inuscules + suppression de la ponctuation, pour les mettresur un pied d'egalite.

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def evaluate():
    wer_scores = []
    y_true, y_pred = [], []

    print(f"{'Fichier':25s} | {'WER':>7s} | {'Attendu':10s} | {'Prédit':10s}")
    print("-" * 65)

    for sample in EVAL_DATASET:
        audio_array, sr = preprocess_audio(sample["file"])
        hypothesis = transcribe(audio_array, sr)

        ref_norm = normalize_text(sample["reference_text"])
        hyp_norm = normalize_text(hypothesis)
        wer = jiwer.wer(ref_norm, hyp_norm) #jiwer calcule le WER entre la reference et l'hypothese normalisées
        wer_scores.append(wer)

        sentiment_result = analyze_sentiment(hypothesis)
        predicted = sentiment_result["label"]

        y_true.append(sample["expected_sentiment"])
        y_pred.append(predicted)

        filename = sample["file"].split("/")[-1]
        print(f"{filename:25s} | {wer:6.1%} | {sample['expected_sentiment']:10s} | {predicted:10s}")

    avg_wer = sum(wer_scores) / len(wer_scores)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")  # average macro calecule le F1 pour chaque classe et fait la moyenne, traite chaque classe de manière égale, même si certaines classes ont moins d'exemples que d'autres.

    print("\n--- Résultats globaux ---")
    print(f"WER moyen (ASR)            : {avg_wer:.1%}")
    print(f"Accuracy (sentiment)       : {accuracy:.1%}")
    print(f"F1-score macro (sentiment) : {f1:.1%}")
    print("\nRapport détaillé par classe :")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    evaluate()