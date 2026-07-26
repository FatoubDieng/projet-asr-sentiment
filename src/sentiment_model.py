from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "cmarkea/distilcamembert-base-sentiment"

# Chargement une seule fois (au démarrage de l'app)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def analyze_sentiment(text):
    """
    Prend un texte (la transcription ASR) et retourne :
    - label : "positif", "négatif" ou "neutre"
    - score : la confiance associée (entre 0 et 1)
    """
    # 1. Vérification que le texte n'est pas vide
    if not text or not text.strip():
        raise ValueError("Le texte à analyser est vide (transcription vide).")

    # 2. Tokenisation du texte
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # 3. Inférence
    with torch.no_grad():
        logits = model(**inputs).logits

    # 4. Softmax pour obtenir des probabilités
    probabilities = torch.softmax(logits, dim=-1)

    # 5. Récupération de la classe prédite et de sa confiance
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    confidence_score = probabilities[0][predicted_class_id].item()

    # 6. Conversion de l'ID en label lisible
    predicted_label = model.config.id2label[predicted_class_id]

    # 7. Normalisation vers les 3 classes principales : "positif", "négatif", "neutre"
    label_mapping = {
        "1 star": "négatif",
        "2 stars": "négatif",
        "3 stars": "neutre",
        "4 stars": "positif",
        "5 stars": "positif",
    }
    final_label = label_mapping.get(predicted_label, predicted_label)

    return {
        "label": final_label,
        "score": round(confidence_score, 4)
    }