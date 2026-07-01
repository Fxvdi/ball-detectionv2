# Ball Detection v2

Projekt im Rahmen des Domänenprojekts I. Ziel ist die automatische Erkennung des Balls in Videomaterial von Handballspielen mittels eines Objekterkennungsmodells (YOLO).

## Ziel

Aus Spielvideos (Handball-Bundesliga-Ausschnitte) soll die Ballposition Frame für Frame erkannt werden, um darauf aufbauend z. B. Flugkurven oder Bewegungsmuster analysieren zu können.

## Vorgehen

- **Datengrundlage:** Frames aus echten Spielmitschnitten (u. a. Stuttgart–Kiel, Stuttgart–Magdeburg, Langenlois) wurden extrahiert und über Roboflow als einzelne Klasse `Ball-Detection` annotiert.
- **Modell:** Ein YOLO-Modell (Ultralytics, Varianten YOLO26m/YOLO26x) wurde auf diesem Datensatz per Transfer Learning/Fine-Tuning auf die Ballerkennung spezialisiert.
- **Training:** Mehrere Trainingsläufe mit unterschiedlichen Modellgrößen, Auflösungen und Trainingsparametern (siehe `yolo/train.ipynb`, `yolo/test_new.ipynb`, `yolo/finetune.py`).
- **Auswertung:** Das fine-getunte Modell wird auf Testvideos angewendet; erkannte Ballpositionen werden pro Frame extrahiert und als Trajektorie (x/y-Koordinaten über die Zeit) visualisiert (`yolo/test.ipynb`, `yolo/test_finetuned.py`).

## Projektstruktur

```
yolo/
  train.ipynb          Training/Fine-Tuning des Detektionsmodells
  finetune.py           Skript für Fine-Tuning inkl. anschließender Inferenz
  test_new.ipynb        Weiterer Trainingslauf (kleinere Auflösung)
  test.ipynb             Inferenz auf Testvideo + Trajektorienanalyse
  test_finetuned.py     Inferenz mit dem fine-getunten Modell
train/                  Annotierter Trainingsdatensatz (nicht im Repo enthalten)
source/                 Beispiel-/Testvideos (nicht im Repo enthalten)
```

## Hinweis zu Daten & Modellgewichten

Trainingsdaten, Rohvideos und trainierte Modellgewichte sind aufgrund ihrer Größe nicht Teil dieses Repositories (siehe `.gitignore`). Versioniert ist ausschließlich der Code zur Datenverarbeitung, zum Training und zur Auswertung.

## Tech-Stack

Python, [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), Roboflow (Datenannotation), pandas/matplotlib (Auswertung).
