# Optimisation et Estimation du Flot Optique

> **English summary.** M1 project (Sorbonne Université, UM4MA266) on estimating optical flow, the dense velocity field between consecutive video frames. It implements and compares the global variational Horn–Schunck method, Total Variation models (TV-L1 / TV-L2, following Chambolle and Sánchez–Meinhardt–Facciolo) and the local Lucas–Kanade method. OpenCV's built-in algorithms serve as references. The scripts study how the regularisation weight α affects the recovered velocity fields and benchmark solver convergence time. The full write-up is in French: [`UgwonaliNathan_OpticalFlow_Rapport.pdf`](UgwonaliNathan_OpticalFlow_Rapport.pdf).
>
> Install: `pip install numpy opencv-python scikit-image matplotlib tqdm`

Ce dépôt contient le code source et les rapports associés à un projet d'étude (niveau M1) portant sur l'estimation du flot optique. Il compare différentes approches mathématiques et algorithmiques, notamment la méthode de Horn-Schunck et les modèles basés sur la Variation Totale (TV-L1 / TV-L2).

## Structure du projet

Le dépôt est organisé de la manière suivante :

* `core/` : Modules fondamentaux de l'algorithme (traitement vidéo, calcul des gradients, solveurs d'optimisation).
* `UgwonaliNathan_RenduFinal/` et `UgwonaliNathan_RenduIntermediaire/` : Dossiers de soumission des livrables.
* `alpha_tests.py` / `alpha_tests_TVL2.py` : Scripts permettant de tester l'influence de l'hyper-paramètre de régularisation (alpha) sur les champs de vitesses.
* `images_extract.py` / `images_extract2.py` : Outils d'extraction de trames synchronisées pour la comparaison visuelle des modèles.
* `time_benchmark.py` / `time_benchmark2.py` : Scripts dédiés au profilage temporel mesurant le temps de convergence des algorithmes.
* `lucas_kanade.py` : Implémentation et tests de la méthode locale de Lucas-Kanade.
* `references_from_opencv.py` : Script appelant les algorithmes natifs de la bibliothèque OpenCV pour établir un point de référence.
* Fichiers `.pdf` : Articles de recherche fondateurs (Horn & Schunck, Chambolle, etc.) et les rapports d'analyse du projet.

## Prérequis

Les algorithmes sont développés en Python. Les bibliothèques suivantes sont nécessaires pour exécuter les scripts :

* numpy
* opencv-python (cv2)
* scikit-image
* matplotlib
* tqdm

L'installation des dépendances s'effectue via l'outil pip :

```bash
pip install numpy opencv-python scikit-image matplotlib tqdm
```
