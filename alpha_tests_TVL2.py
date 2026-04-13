# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:22:16 2026

@author: ugwna
Script d'expérimentation : Flot Optique par Variation Totale (TV-L1 / ROF)
"""

import os
import cv2
import numpy as np
import sys
# import matplotlib.pyplot as plt

# --- AJUSTE CE CHEMIN SELON LA RACINE DE TON PROJET ---
project_root = r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet"
sys.path.append(project_root)

from core.video_io import VideoDataHandler
from core.gradients import GradIntensity
from core.velocity import Velocity
from core.optimizer import Optimization

def calculate_energies(u, v, Ix, Iy, It):
    """Calcule l'énergie d'attache aux données et l'énergie de lissage (Variation Totale)."""
    # Énergie d'attache aux données
    e_data = np.mean((Ix * u + Iy * v + It)**2)
    
    # Énergie de lissage (Norme L1 des gradients = Variation Totale)
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    # On utilise la valeur absolue (L1) et non le carré !
    e_smooth = np.mean(np.abs(du_dx) + np.abs(du_dy) + np.abs(dv_dx) + np.abs(dv_dy))
    
    return e_data, e_smooth

def draw_quiver(frame, u, v, step=20, scale=10, color=(0, 255, 0)):
    """Dessine les vecteurs de mouvement sur l'image originale."""
    img = frame.copy()
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    
    for xi, yi in zip(x, y):
        dx, dy = int(u[yi, xi] * -scale), int(v[yi, xi] * -scale)
        if abs(dx) > 1 or abs(dy) > 1: 
            cv2.arrowedLine(img, (xi, yi), (xi + dx, yi + dy), color, 1, tipLength=0.3)
    return img

def run_experiment_tvl2(video_path, alpha_list):
    base_name = os.path.basename(video_path)
    file_name = os.path.splitext(base_name)[0]
    
    results_data = []
    results_smooth = []

    for alpha_val in alpha_list:
        print(f"\n>>> Traitement TV-L1 : {file_name} | Alpha : {alpha_val}")
        
        # Dossier spécifique pour le TV-L1
        output_dir = os.path.join(project_root, "results_TVL2", file_name, f"alpha_{alpha_val}")
        os.makedirs(output_dir, exist_ok=True)
        
        path_hsv = os.path.join(output_dir, f"{file_name}_hsv.avi")
        path_vec = os.path.join(output_dir, f"{file_name}_vector.avi")

        handler = VideoDataHandler(video_path)
        prev_frame = handler.get_initial_frame() 
        
        velocity = Velocity(shape=prev_frame.shape)
        optimizer = Optimization(alpha=alpha_val)
        grad_engine = GradIntensity()
        
        fps = handler.cap.get(cv2.CAP_PROP_FPS) or 30
        h, w = prev_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer_hsv = cv2.VideoWriter(path_hsv, fourcc, fps, (w, h))
        writer_vec = cv2.VideoWriter(path_vec, fourcc, fps, (w, h))
        
        if not writer_hsv.isOpened():
            print("ERREUR CRITIQUE : VideoWriter n'a pas pu s'initialiser.")
            return

        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        frame_e_data = []
        frame_e_smooth = []
        
        try:
            for p_frame, c_frame, raw_frame in handler.stream_frames():
                
                # ==========================================================
                # CORRECTION : RAZ du champ de vitesses pour chaque frame
                # ==========================================================
                velocity.u.fill(0.0)
                velocity.v.fill(0.0)
                
                # Calcul des gradients
                grad_engine.Ex, grad_engine.Ey, grad_engine.Et = grad_engine.compute(p_frame, c_frame)
                
                # Appel de l'optimiseur TV-L1
                optimizer.ForwardBackwardTVL2(grad_engine, velocity, nit=100, tv_iter=25, tol=5e-4)
        
                # Calcul des énergies
                e_d, e_s = calculate_energies(velocity.u, velocity.v, 
                                              grad_engine.Ex, grad_engine.Ey, grad_engine.Et)
                frame_e_data.append(e_d)
                frame_e_smooth.append(e_s)
                
                # Visualisation HSV
                mag, ang = cv2.cartToPolar(velocity.u, velocity.v)
                hsv[..., 0] = ang * 180 / np.pi / 2 
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                writer_hsv.write(rgb_flow)
                
                # Visualisation Vecteurs
                vector_frame = draw_quiver(raw_frame, velocity.u, velocity.v, step=20, scale=12)
                writer_vec.write(vector_frame)
                
                cv2.imshow("Flux HSV (TV-L1)", rgb_flow)
                cv2.imshow("Champ de Vecteurs (TV-L1)", vector_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break 

        finally:
            handler.release()
            writer_hsv.release()
            writer_vec.release()
            cv2.destroyAllWindows()
            
        # Moyenne des énergies pour l'Alpha courant
        avg_e_data = np.mean(frame_e_data)
        avg_e_smooth = np.mean(frame_e_smooth)
        results_data.append(avg_e_data)
        results_smooth.append(avg_e_smooth)
        print(f"Bilan Alpha {alpha_val} -> E_data: {avg_e_data:.4f} | E_smooth (TV): {avg_e_smooth:.4f}")

    return alpha_list, results_data, results_smooth

if __name__ == "__main__":
    V_PATH = r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet\data\360p_RigidBodyMovement.mp4"
    
    # Plage d'Alphas adaptée pour la Variation Totale (souvent < 1.0)
    ALPHAS = [1e-2, 1e0, 1e2]
    alphas_tested, e_data, e_smooth = run_experiment_tvl2(V_PATH, ALPHAS)
    
    # # --- TRACÉ DU GRAPHIQUE ---
    # plt.figure(figsize=(10, 6))
    # plt.plot(e_data, e_smooth, marker='s', linestyle='-', color='r', linewidth=2, label="Méthode TV-L1")
    
    # for i, alpha in enumerate(alphas_tested):
    #     plt.annotate(f"$\\alpha={alpha}$", (e_data[i], e_smooth[i]), 
    #                  textcoords="offset points", xytext=(10,5), ha='left')

    # plt.title("Compromis Attache aux données / Régularisation L1 (Modèle ROF)")
    # plt.xlabel("Énergie d'attache aux données ($E_{data}$)")
    # plt.ylabel("Énergie de Variation Totale ($E_{smooth} L^1$)")
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    
    # plt.savefig(os.path.join(project_root, "results_TVL2", "Tradeoff_Alpha_TVL2.png"), dpi=300, bbox_inches='tight')
    # plt.show()
