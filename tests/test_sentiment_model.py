import pytest
from src.sentiment_model import analyze_sentiment


def test_sentiment_positive_text():
    """Un texte clairement positif doit être classé 'positif'."""
    result = analyze_sentiment("C'était un excellent produit, je suis ravi !")
    assert result["label"] == "positif"
    assert 0.0 <= result["score"] <= 1.0


def test_sentiment_negative_text():
    """Un texte clairement négatif doit être classé 'négatif'."""
    result = analyze_sentiment("C'était vraiment horrible, je suis très déçu.")
    assert result["label"] == "négatif"
    assert 0.0 <= result["score"] <= 1.0


def test_sentiment_empty_text_raises_error():
    """Un texte vide doit lever une ValueError claire."""
    with pytest.raises(ValueError):
        analyze_sentiment("")


def test_sentiment_output_keys():
    """Le résultat doit toujours contenir les clés 'label' et 'score'."""
    result = analyze_sentiment("Le film était correct.")
    assert "label" in result
    assert "score" in result