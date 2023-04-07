# UV-Project Integration of RealSense scene interpretation

Nous sommes Rebergues Alexis et Léa Costa, deux étudiants à l'IMT Nord Europe. Dans le cadre de notre formation, nous avons été amené à réaliser un projet d'un mois.
Celui-ci est basé sur YoloV7 et permet à l'utilisateur qui le souhaite d'utiliser une caméra Realsense afin de faire de la détection d'objet dans un environnement parmis 80 objets. Les objets détectés seront détourés ( segmentation d'instance ), et les distances entre la caméra et les objets seront affichées.
Le tracking des objets ainsi que l'enregistrement des captures d'images sont également disponibles.

Ce programme peut être déployé sur un robot, le processus pouvant fonctionner sous ROS2. L'ensemble de la documentation de ROS2 est disponible [ici](https://docs.ros.org/en/foxy/index.html).

Une vidéo illutrant ce projet est disponible [ici](https://youtu.be/K1XV7OQ4Z0g)

Ce projet a été réalisé sur Linux Ubuntu 20.04 et la compatibilité avec d'autres systèmes d'exploitation n'est pas garantie.

## Etapes à suivre pour lancer le code 

L'installation est basée sur le git disponible [ici](https://github.com/RizwanMunawar/yolov7-segmentation/)

Se placer dans le dossier ros2_ws : 

` cd ros2_ws `

Cloner le dépôt disponible au lien donné précédemment :

` git clone https://github.com/RizwanMunawar/yolov7-segmentation.git `

Se placer dans le dossier crée :

` cd yolov7-segmentation `

Remplacer les fichiers segment/predict.py et utils/dataloader.py par ceux de ce dépôt <!-- En 1 phrase pourquoi ?  -->

Créer un environnement virtuel (sur Linux) : 

```sh
python3 -m venv yolov7seg
source yolov7seg/bin/activate 
```
Mettre à jour pip :

` pip install --upgrade pip `

Installer requirements :

` pip install -r requirements.txt `

Télécharger le fichier weights en suivant le lien suivant : [https://github.com/RizwanMunawar/yolov7-segmentation/releases/download/yolov7-segmentation/yolov7-seg.pt](https://github.com/RizwanMunawar/yolov7-segmentation/releases/download/yolov7-segmentation/yolov7-seg.pt)  et le placer dans le dossier "yolov7-segmentation"

Installer pyrealsense2:

`pip install pyrealsense2`

Lancer le programme en exécutant : 

`python3 segment/predict.py`

## Présentation de la solution

Le programme proposé fonctionne en temps réel sur caméra RealSense. Il est capable de reconnaître 80 classes d'objets. Les objets reconnus sont ensuites placés dans des boîtes englobantes et le programme réalise une segmentation d'objets. L'option tracking, qui peut être desactivée dans le fichier predict.py permet de suivre la trajectoire du centre de la boîte de chaque objet. Après cela, la position du centre de gravité de l'objet est calculée et la distance entre ce point et la caméra est renvoyée.

### YoloV7 + segmentation

<!-- Présentation succinth de cette solution, ces bases (torch) ; mise en parrallel des concurence-->
Le programme s'appuie sur la bibliothèque torch YOLOv7 qui, grâce à des modèles pré-entraînés, est capable de reconnaître 80 classes d'objets, les placer dans des boîtes englobantes et faire de la segmentation d'objets.

### Appropriation de l'outil

<!-- Modification que vous avez opéré pour vous approprié l'outils sur votre problématique. Pointeur vers le code ou on peut trouver l'éléments -->

1. 
2. **Modif-2** ...
1. **Modif-3** ...
1. **Modif-4** ...


### Intégration dans ROS2 




## Perspectives d'améliorations 

* Le programme tourne actuellement sur CPU et a donc une vitesse limitée. Une amélioration possible serait de le faire tourner sur GPU notamment en utilisant CUDA et en modifiant le paramètre "device" lors du lancement du code.

* Déplacer le robot dans un environnement et utiliser la reconnaissance d'objets pour créer une carte en temps réel en y ajoutant les objets detectés au fur et à mesure.

* Les objets qui ne sont pas des humains sont moins mobiles que ces-derniers. Ainsi, une perspective d'amélioration de la vitesse serait de ne les rechercher qu'à une certaine fréquence et pas pour chaque frame reçue

* Ne pas utiliser YOLO mais faire un apprentissage afin de pouvoir avoir un programme plus léger qui reconnait plus d'objets de l'environnement ciblé et moins d'autres objets. De plus, refaire un apprentissage pourrait permettre au robot de reconnaitre des humains individuellement.
