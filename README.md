# Introduction

Nous sommes Rebergues Alexis et Léa Costa, deux étudiants à l'IMT Nord Europe. Dans le cadre de notre formation, nous avons été amené à réaliser un projet d'un mois.
Celui-ci est basé sur YoloV7 et permet à l'utilisateur qui le souhaite d'utiliser une caméra Realsense afin de faire de la détection d'objet dans un environnement parmis 80 objets. Les objets détectés seront détourés ( segmentation d'instance ), et les distances entre la caméra et les objets seront affichées.
Le tracking des objets ainsi que l'enregistrement des captures d'images sont également disponibles.

Ce programme peut être déployé sur un robot, le processus pouvant fonctionner sous ROS2. L'ensemble de la documentation de ROS2 est disponible via le lien suivant: https://docs.ros.org/en/foxy/index.html .

# Etapes à suivre pour lancer le code 

L'installation est basée sur le git disponible au lien suivant : https://github.com/RizwanMunawar/yolov7-segmentation/

Se placer dans le dossier ros2_ws : 

` cd ros2_ws `

Cloner le dépôt disponible au lien donné précédemment :

` git clone https://github.com/RizwanMunawar/yolov7-segmentation.git `

Se placer dans le dossier crée :

` cd yolov7-segmentation `

Remplacer les fichiers segment/predict.py et utils/dataloader.py par ceux de ce dépôt 

Créer un environnement virtuel (sur Linux) : 
```
python3 -m venv yolov7seg
source yolov7seg/bin/activate 
```
Mettre à jour pip :

` pip install --upgrade pip `

Installer requirements :

` pip install -r requirements.txt `

Télécharger le fichier weights en suivant le lien suivant : https://github.com/RizwanMunawar/yolov7-segmentation/releases/download/yolov7-segmentation/yolov7-seg.pt  et le placer dans le dossier "yolov7-segmentation"

Lancer le programme en exécutant : 

`python3 segment/predict.py`

# Explications 

# Perspectives d'améliorations 


* Le programme tourne actuellement sur CPU et a donc une vitesse limitée. Une amélioration possible serait de le faire tourner sur GPU notamment en utilisant CUDA et en modifiant le paramètre "device" lors du lancement du code.

* Déplacer le robot dans un environnement et utiliser la reconnaissance d'objets pour créer une carte en temps réel en y ajoutant les objets detectés au fur et à mesure.

* Les objets qui ne sont pas des humains sont moins mobiles que ces-derniers. Ainsi, une perspective d'amélioration de la vitesse serait de ne les rechercher qu'à une certaine fréquence et pas pour chaque frame reçue

* Ne pas utiliser YOLO mais faire un apprentissage afin de pouvoir avoir un programme plus léger qui reconnait plus d'objets de l'environnement ciblé et moins d'autres objets. De plus, refaire un apprentissage pourrait permettre au robot de reconnaitre des humains individuellement.
