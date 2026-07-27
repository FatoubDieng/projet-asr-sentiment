# Image de base légère avec Python 3.11 préinstallé
FROM python:3.11-slim

# Installation de ffmpeg (nécessaire pour décoder les fichiers .mp3 via librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail à l'intérieur du conteneur
WORKDIR /app

# Copie et installation des dépendances Python en premier
COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code de l'application
COPY . .

# Port exposé par l'API FastAPI
EXPOSE 8000

##Commande exécutée au démarrage du conteneur
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

