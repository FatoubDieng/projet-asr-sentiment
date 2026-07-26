import numpy as np
import pytest
from src.audio_preprocessing import preprocess_audio


def test_preprocess_valid_file():
    """Un fichier valide doit retourner un array mono à 16kHz."""
    audio_array, sample_rate = preprocess_audio("data/samples/exemple_test.wav")
    assert sample_rate == 16000
    assert audio_array.ndim == 1  # signal mono = 1 dimension


def test_preprocess_missing_file():
    """Un chemin de fichier inexistant doit lever une ValueError claire."""
    with pytest.raises(ValueError):
        preprocess_audio("chemin/qui/nexiste/pas.wav")


def test_preprocess_normalization():
    """Après normalisation, la moyenne doit être proche de 0."""
    audio_array, _ = preprocess_audio("data/samples/exemple_test.wav")
    assert abs(np.mean(audio_array)) < 0.1