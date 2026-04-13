# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:16:13 2026

@author: ugwna
"""

import cv2
import numpy as np

def lucas_kanade_sparse(video_path, max_corners=100):
    """
    Flot optique sparse avec Lucas-Kanade.
    Suit des points d'intérêt entre frames successives.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Paramètres pour la détection de coins
    feature_params = dict(
        maxCorners=max_corners,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # Paramètres Lucas-Kanade
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    ret, old_frame = cap.read()
    if not ret:
        cap.release()
        return []
        
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
        
        # Sélection des bons points
        if p1 is not None:
            good_new = p1[status == 1]
            good_old = p0[status == 1]
            
            # Stockage des déplacements
            for new, old in zip(good_new, good_old):
                trajectories.append({
                    'old': old.flatten(),
                    'new': new.flatten(),
                    'flow': (new - old).flatten()
                })
                
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            
        else:
            # Si LK a perdu tous les points, on arrête
            break
            
    cap.release()
    return trajectories


def farneback_dense(frame1, frame2):
    """
    Flot optique dense avec méthode de Farneback.
    Retourne le champ de flot (u, v) pour chaque pixel.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 
        pyr_scale=0.5,    # facteur de réduction pyramide
        levels=3,         # nombre de niveaux
        winsize=15,       # taille fenêtre
        iterations=3,     # itérations par niveau
        poly_n=5,         # taille voisinage polynôme
        poly_sigma=1.2,   # sigma du gaussien
        flags=0
    )
    
    # flow[:,:,0] = composante u (horizontale)
    # flow[:,:,1] = composante v (verticale)
    return flow


def visualize_flow(flow):
    """
    Visualise le flot optique en HSV (teinte=direction, saturation=magnitude).
    """
    magnitude, angle = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    
    hsv[..., 0] = angle * 180 / np.pi / 2                  # Teinte (Direction)
    hsv[..., 1] = 255                                      # Saturation (Maximale)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) # Valeur (Vitesse)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

video_path = r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet\data\360p_F1.mp4"
flow = lucas_kanade_sparse(video_path)
flow = farneback_dense(frame1, frame2)
image_resultat = visualize_flow(flow)

cv2.imshow("Flot Optique - Farneback", image_resultat)
cv2.waitKey(30) # Pause de 30ms pour voir l'image