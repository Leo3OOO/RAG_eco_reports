### 📦 Benutzung

---

### 🛠️ Schritte zum Funktionieren

1. **Repository klonen**  
Klone unser Git-Repository in einen gewünschten Ordner und navigiere anschließend in diesen Ordner:

  ```bash 
  cd "folder_name" 
  ```

2.	**Setup (für Mac oder Linux)**
Führe im Terminal (im gewählten Ordner) folgende Befehle aus:

	```bash
	chmod +x setup_env.sh
	``` 

	```bash
	./setup_env.sh
	```

	```bash
	newenv/bin/activate
	```


   > **Hinweis:** Falls du PDFs verarbeiten möchtest, installiere Poppler lokal (nur notwendig bei Nutzung ohne Docker):

   * **Debian/Ubuntu:**

     ```bash
     sudo apt-get update
     sudo apt-get install poppler-utils
     ```
   * **MacOS (Homebrew):**

     ```bash
     brew install poppler
     ```


3. **.env-Datei erstellen**
Erstelle eine Datei mit dem Namen ```.env``` und füge den API-Key wie folgt ein:

```API_KEY='dein Eintrag'```

Speichere anschließend die Datei.

---

### 🚀 Schritte zum Ausführen der Datei (aktuelle Nutzung)
	1.	Im Terminal zu dem Passenden Ordner navigieren
	2.	Folgenden Befehl im Terminal ausführen:

	```bash 
	streamlit run main.py
	```

### 🔁 Wiederholte Benutzung

Wenn du das Projekt erneut öffnen möchtest, musst du das Virtual Environment neu aktivieren:
	1.	Terminal in den passenden Ordner leiten
	2.	Virtual Environment aktivieren und Code ausführen:

	```bash 
	source newenv/bin/activate
	streamlit run main.py
	```

### 🐳 Nutzung mit Docker

Falls du Docker nutzen möchtest, kannst du das Projekt containerisiert ausführen.

#### Voraussetzungen

* Docker
* Docker Compose

#### Schritte

1. **Docker Image bauen und starten**

   Im Projektordner ausführen:

   ```bash
   docker-compose up --build
   ```

   Dadurch wird:

   * das Image erstellt,
   * der Container gestartet,
   * die App automatisch ausgeführt.

2. **Streamlit App aufrufen**

   Öffne deinen Browser und gehe zu:

   ```
   http://localhost:8080
   ```

   Dort sollte die Anwendung sichtbar sein.

3. **Container beenden**

   Zum Beenden drücke `Ctrl + C` und fahre die Container herunter:

   ```bash
   docker-compose down
   ```

---

--- 

### 🗂️ Trello

Link to the Trello Board

https://trello.com/b/8bUm8R9D/ai-application-project

