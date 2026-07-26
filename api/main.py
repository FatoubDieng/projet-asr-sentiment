import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.pipeline import run_pipeline

app = FastAPI(
    title="API ASR + Analyse de sentiment",
    description="Transcrit un fichier audio et analyse le sentiment exprimé.",
    version="1.0.0",
)

ALLOWED_EXTENSIONS = {".wav", ".mp3"}


@app.get("/")
def health_check():
    """Endpoint simple pour vérifier que l'API est en ligne."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit un fichier audio, retourne la transcription, le sentiment
    et le score de confiance.
    """
    # 1. Validation de l'extension du fichier
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {ext}. Formats acceptés : .wav, .mp3",
        )

    # 2. Sauvegarde temporaire du fichier reçu sur le disque
    #    (nécessaire car torchaudio.load() attend un chemin de fichier)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Le fichier envoyé est vide.")
        tmp_file.write(content)
        tmp_path = tmp_file.name

    # 3. Exécution du pipeline avec gestion d'erreurs
    try:
        result = run_pipeline(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur interne du serveur : {e}"
        )
    finally:
        # 4. Nettoyage : suppression du fichier temporaire dans tous les cas
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result