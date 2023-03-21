# Yolooooo

because you only live once 🧀

## Pour le groupe qui s'occupe d'Openpose

    mkdir images
    mkdir runs/detect

Il faut juste lancer le `main.py`.

### Fonctionnement du programme

- La vidéo `mathis_squat.mp4` est lue par le programme, puis découpé en images qui sont enregistrées dans le dossier `images`
- Chaque image dans le dossier `images` passe par la détection yolov5, puis les objets détéctés sont enregistrés dans le dossier `runs/detect`
- Dans notre cas, les images croppées qui nous intéressent seront enregistrées dans `runs/detect/exp*/person`