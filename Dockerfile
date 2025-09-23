# Image de base légère avec Python et TensorFlow
FROM python:3.11-slim

# Installer quelques dépendances système nécessaires pour numpy, pandas, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /app

# Copier requirements et installer
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code, y compris ton package local informer
COPY . .

# Exposer le port
EXPOSE 5000

# Commande de lancement (gunicorn pour production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:server"]

