# Image de base avec Python 3.11 
FROM python:3.11-slim

# Installation de ffmpeg (Pour décoder les fichiers .mp3 via librosa ou torchaudio (qui a une erreur de notre cote ))
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail à l'intérieur du conteneur
WORKDIR /app

# Copie et installation des dépendances Python
# (permet de réutiliser ce cache Docker si seul le code change, pas les dépendances)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code de l'application
COPY . .

# Port exposé par l'API FastAPI
EXPOSE 8000

# Commande exécutée au démarrage du conteneur
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]