import pytest
from src.audio_preprocessing import preprocess_audio
from src.asr_model import transcribe


def test_transcribe_returns_string():
    """La transcription doit être une chaîne de caractères non vide."""
    audio_array, sample_rate = preprocess_audio("data/samples/exemple_test.wav")
    result = transcribe(audio_array, sample_rate)
    assert isinstance(result, str)


def test_transcribe_not_empty_on_valid_audio():
    """Un audio contenant de la parole doit produire une transcription non vide."""
    audio_array, sample_rate = preprocess_audio("data/samples/exemple_test.wav")
    result = transcribe(audio_array, sample_rate)
    assert len(result.strip()) > 0\
    