# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:27:00 2026

@author: ugwna
"""
import cv2
import sys
import os 
sys.path.append(r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet")


def lucas_kanade_sparse(video_path, max_corners=10):
    """
    Flot optique sparse avec Lucas-Kanade.
    Suit des points d'interet entre frames successives.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Parametres pour la detection de coins
    feature_params_F1 = dict(
    maxCorners=100,        
    qualityLevel=0.1,      # Très tolérant pour accrocher la carrosserie
    minDistance=15,        # Force les points à se répartir sur toute la voiture
    blockSize=5            # Loupe plus petite pour les détails de la monoplace
    )
    
    feature_params_rigid = dict(
    maxCorners=100,       
    qualityLevel=0.2,      # Assez strict pour ne garder que les points parfaits
    minDistance=30,        # Grande distance pour un bel affichage aéré des flèches
    blockSize=7            # Fenêtre standard
    )
    
    feature_params = feature_params_rigid
    
    # Parametres Lucas-Kanade
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    trajectories = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calcul du flot optique
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        
        # Selection des bons points
        if p1 is not None:
            good_new = p1[status == 1]
            good_old = p0[status == 1]
            
            # Stockage des deplacements
            for new, old in zip(good_new, good_old):
                trajectories.append({
                    'old': old.flatten(),
                    'new': new.flatten(),
                    'flow': (new - old).flatten()
                })
                
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            
    cap.release()
    return trajectories


def draw_sparse_flow(base_image, trajectories, scale=1):
    """
    Dessine les vecteurs de mouvement épars sur une image de fond.
    """
    vis_img = base_image.copy()
    
    for t in trajectories:
        # Récupération et conversion en entiers des coordonnées 
        x_old, y_old = int(t['old'][0]), int(t['old'][1])
        x_new, y_new = int(t['new'][0]), int(t['new'][1])
        
        # Application de l'échelle si le mouvement est trop petit pour être bien visible
        if scale != 1:
            dx, dy = t['flow']
            x_new = int(x_old + dx * scale)
            y_new = int(y_old + dy * scale)
        
        # 1. Dessiner la flèche de mouvement (en rouge)
        cv2.arrowedLine(vis_img, (x_old, y_old), (x_new, y_new), 
                        color=(0, 0, 255), thickness=1, tipLength=0.3)
        
        # 2. Dessiner l'origine du point (petit cercle vert)
        cv2.circle(vis_img, (x_old, y_old), radius=2, color=(0, 255, 0), thickness=-1)
        
    return vis_img



## Test de l'algorithme de Lucas-Kanade
V_PATH = r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet\data\360p_RigidBodyMovement.mp4"
trajectories = lucas_kanade_sparse(V_PATH)

## 1. Lire la première frame pour s'en servir de fond
cap = cv2.VideoCapture(V_PATH)
ret, frame_fond = cap.read()
cap.release()

## 2. Dessiner et sauvegarder 
image_resultat = draw_sparse_flow(frame_fond, trajectories, scale=5)
project_root = r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet\results_LK"
os.makedirs(project_root, exist_ok=True)
base_name = os.path.basename(V_PATH)  # Isole "360p_F1.mp4"
name, _ = os.path.splitext(base_name)     # Sépare "360p_F1" et ".mp4"
chemin_final = os.path.join(project_root, f"LK_sparse_flow_improved_{name}.png")

if image_resultat is None:
    print("Erreur : L'image à sauvegarder est vide (None).")
else:
    is_success, im_buf_arr = cv2.imencode(".png", image_resultat)
    
    if is_success:
        im_buf_arr.tofile(chemin_final)
        print(f"Succès ! Image sauvegardée dans : {chemin_final}")
    else:
        print("Erreur lors de l'encodage de l'image.")