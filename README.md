
### Benutzung 

### Schritte zum Funktionieren: 

1. Clone unser Git-Reposetory in einen von die gewählten Ordner und navigiere in den erstellten Ordner mit cd "folder_name"

2. Bei Mac oder Linux: Folgenden Befehl ins Terminal(geleitet zum gewählten Ordner) eingeben: 

chmod +x setup_env.sh
./setup_env.sh
source newenv/bin/activate

2. Eine Datei mit dem Namen ".env" erstellen und den passenden API key reinschreiben
   
API_KEY='dein Eintag'

 -> dann muss diese Datei gespeichert werden

### Schritte zum ausführen der Datei im Moment: 

Terminal in den passenden Ordner leiten 

in das Terminal schreiben: 

python working_backend.py

### Wiederholte Benutzung

### Wenn man es wieder öffnet muss das Virtual Environment neu aktiviert werden, dies geht mit: 

1. Terminal in den passenden Ordner leiten 

2. In das Terminal schreiben (aktivieren des Virtual Environment + Code ausführen): 

source newenv/bin/activate
python working_backend.py


