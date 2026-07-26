import gradio as gr
from src.pipeline import run_pipeline


def process_audio_ui(audio_file):
    """
    Fonction appelée par Gradio à chaque soumission d'un fichier audio.
    Gradio nous donne directement un chemin de fichier temporaire.
    """
    if audio_file is None:
        return "Aucun fichier fourni.", "-", "-"

    try:
        result = run_pipeline(audio_file)
        return (
            result["transcription"],
            result["sentiment"],
            f"{result['confidence'] * 100:.1f} %",
        )
    except ValueError as e:
        return f"Erreur : {e}", "-", "-"
    except Exception as e:
        return f"Erreur interne : {e}", "-", "-"


demo = gr.Interface(
    fn=process_audio_ui,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath"),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Score de confiance"),
    ],
    title="ASR + Analyse de sentiment",
    description=(
        "Envoyez un fichier audio (.wav ou .mp3, max 5 min) pour obtenir "
        "sa transcription et le sentiment exprimé (positif / négatif / neutre)."
    ),
)

if __name__ == "__main__":
    demo.launch()