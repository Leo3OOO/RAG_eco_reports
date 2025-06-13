""" Entwerder folgenden Befehl ins Terminal eingeben: 

chmod +x setup_env.sh
./setup_env.sh

"""



#!/bin/bash

# Name des Environments
ENV_NAME="newenv"

# Virtual Environment erstellen, falls nicht existiert
if [ ! -d "$ENV_NAME" ]; then
  python3 -m venv $ENV_NAME
  echo "Virtual Environment $ENV_NAME wurde erstellt."
else
  echo "Virtual Environment $ENV_NAME existiert bereits."
fi

# Environment aktivieren (für Bash/Mac/Linux)
source $ENV_NAME/bin/activate

# pip updaten und requirements installieren
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup abgeschlossen. Environment $ENV_NAME ist aktiviert."
