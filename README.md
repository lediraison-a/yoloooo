# Yolooooo

because you only live once 🧀

## Pour lancer une détection en CLI

Pour lancer une détection sur images/mecs.jpg:

    git clone https://github.com/ultralytics/yolov5
    cd yolov5 
    pip install -r requirements.txt
    python detect.py --classes 0 --weights yolov5s.pt --source ../images/mecs.jpg

> Le paramètre `--classes 0` permet de détecter uniquement les personnes.

Les résultats seront générés dans yolov5/runs/detect.

Pour utiliser la webcam:

    python detect.py --classes 0 --source 0

## Pour le groupe qui s'occupe d'Openpose

Il faut :

    git clone https://github.com/ultralytics/yolov5
    cd yolov5 
    pip install -r requirements.txt
    cd ..

Puis lancer le `main.py`.

### Fonctionnement du programme

- La vidéo `mathis_squat.mp4` est lue par le programme, puis découpé en images qui sont enregistrées dans le dossier `images`
- Chaque image dans le dossier `images` passe par la détection yolov5, puis les objets détéctés sont enregistrés dans le dossier `runs/detect`
- Dans notre cas, les images croppées qui nous intéressent seront enregistrées dans `runs/detect/exp*/person`