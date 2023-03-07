# Yolooooo

because you only live once 🧀

## Pour lancer une détection

Pour lancer une détection sur images/mecs.jpg:

    git clone https://github.com/ultralytics/yolov5
    cd yolov5 
    pip install -r requirements.txt
    python detect.py --classes 0 --weights yolov5s.pt --source ../images/mecs.jpg

> Le paramètre `--classes 0` permet de détecter uniquement les personnes.

Les résultats seront générés dans yolov5/runs/detect.

Pour utiliser la webcam:

    python detect.py --classes 0 --source 0