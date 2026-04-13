# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:35:25 2026

@author: ugwna
"""

import os
import cv2
import numpy as np
import sys
sys.path.append(r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet")

from core.video_io import VideoDataHandler
from core.gradients import GradIntensity
from core.velocity import Velocity
from core.optimizer import Optimization

def draw_quiver(frame, u, v, step=20, scale=10, color=(0, 255, 0)):
    """Dessine les vecteurs de mouvement sur l'image originale."""
    img = frame.copy()
    h, w = img.shape[:2]
    # Création d'une grille pour l'affichage des vecteurs
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    
    for xi, yi in zip(x, y):
        # On multiplie par scale pour mettre les flèches à l'endroit
        dx, dy = int(u[yi, xi] * - scale), int(v[yi, xi] * - scale)
        if abs(dx) > 1 or abs(dy) > 1: # Seuil minimal pour éviter de dessiner le bruit
            cv2.arrowedLine(img, (xi, yi), (xi + dx, yi + dy), 
                            color, 1, tipLength=0.3)
    return img

def run_experiment(video_path, alpha_list):
    base_name = os.path.basename(video_path)
    file_name = os.path.splitext(base_name)[0]
    
    # 1. FORCER LE CHEMIN ABSOLU
    project_root = r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet"

    for alpha_val in alpha_list:
        print(f"\n>>> Traitement : {file_name} | Alpha : {alpha_val}")
        
        # Le dossier de sortie sera FORCÉMENT dans ton projet
        output_dir = os.path.join(project_root, "results", file_name, f"alpha_{alpha_val}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. CHANGER L'EXTENSION EN .avi (Optionnel mais recommandé si le mp4 bloque)
        path_hsv = os.path.join(output_dir, f"{file_name}_hsv.avi")
        path_vec = os.path.join(output_dir, f"{file_name}_vector.avi")

        handler = VideoDataHandler(video_path)
        prev_frame = handler.get_initial_frame() 
        
        velocity = Velocity(shape=prev_frame.shape)
        optimizer = Optimization(alpha=alpha_val)
        grad_engine = GradIntensity()
        
        fps = handler.cap.get(cv2.CAP_PROP_FPS) or 30
        h, w = prev_frame.shape[:2]
        
        # 3. UTILISER LE CODEC XVID POUR WINDOWS
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        writer_hsv = cv2.VideoWriter(path_hsv, fourcc, fps, (w, h))
        writer_vec = cv2.VideoWriter(path_vec, fourcc, fps, (w, h))
        
        # Ajout d'une vérification de sécurité
        if not writer_hsv.isOpened():
            print("ERREUR CRITIQUE : VideoWriter n'a pas pu s'initialiser.")
            return

        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255

        try:
            # On suppose que stream_frames() renvoie (p_gray, c_gray, c_raw_color)
            for p_frame, c_frame, raw_frame in handler.stream_frames():
                
                # ==========================================================
                # CORRECTION : RAZ du champ de vitesses pour chaque frame
                # ==========================================================
                velocity.u.fill(0.0)
                velocity.v.fill(0.0)
                
                grad_engine.Ex, grad_engine.Ey, grad_engine.Et = grad_engine.compute(p_frame, c_frame)
                optimizer.GaussSeidel(grad_engine, velocity, nit=100, tol=5e-4)
                
                # --- Visualisation 1 : HSV ---
                mag, ang = cv2.cartToPolar(velocity.u, velocity.v)
                hsv[..., 0] = ang * 180 / np.pi / 2  # Correction ici
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                writer_hsv.write(rgb_flow)
                
                # --- Visualisation 2 : Vecteurs sur Image Originale ---
                vector_frame = draw_quiver(raw_frame, velocity.u, velocity.v, step=20, scale=12)
                writer_vec.write(vector_frame)
                
                # Affichage des deux pour monitoring
                cv2.imshow("Flux HSV", rgb_flow)
                cv2.imshow("Champ de Vecteurs", vector_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return 

        finally:
            handler.release()
            writer_hsv.release()
            writer_vec.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    V_PATH = r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet\data\360p_F1.mp4"
    ALPHAS = [1e4]
    run_experiment(V_PATH, ALPHAS)
    
    
    
    
    
    
    