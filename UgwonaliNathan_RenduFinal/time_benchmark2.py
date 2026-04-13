# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:36:52 2026

@author: ugwna
Script de Benchmark Temporel des algorithmes d'Optimisation
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- AJUSTE CE CHEMIN SELON LA RACINE DE TON PROJET ---
project_root = r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet"
sys.path.append(project_root)

from core.video_io import VideoDataHandler
from core.gradients import GradIntensity
from core.velocity import Velocity
from core.optimizer import Optimization

def run_time_benchmark(video_path, alpha=1.0, nit=100, tol=1e-4):
    handler = VideoDataHandler(video_path)
    
    # Récupération du nombre total de frames pour l'affichage
    total_frames = int(handler.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Lancement du benchmark sur TOUTE la vidéo ({total_frames} frames) : {os.path.basename(video_path)}")
    print(f"Paramètres : alpha={alpha}, itérations max={nit}, tol={tol}\n")

    prev_frame = handler.get_initial_frame()
    h, w = prev_frame.shape[:2]

    grad_engine = GradIntensity()

    # Dictionnaires réduits à 2 méthodes
    methods = ["Gauss-Seidel (Horn-Schunck)", "FB TV-L2 (Chambolle)"]
    times = {m: [] for m in methods}
    
    frame_count = 0
    
    try:
        for p_frame, c_frame, raw_frame in handler.stream_frames():
            frame_count += 1
            print(f"--- Traitement de la Frame {frame_count}/{total_frames} ---")
            
            # Les gradients sont calculés UNE SEULE FOIS
            grad_engine.Ex, grad_engine.Ey, grad_engine.Et = grad_engine.compute(p_frame, c_frame)

            # 1. Test Gauss-Seidel
            vel_gs = Velocity(shape=(h, w))
            vel_gs.u.fill(0.0) ; vel_gs.v.fill(0.0)
            opt_gs = Optimization(alpha=alpha)
            opt_gs.GaussSeidel(grad_engine, vel_gs, nit=nit, tol=tol)
            times["Gauss-Seidel (Horn-Schunck)"].append(opt_gs.convergence_time)

            # 2. Test Forward-Backward TV-L2
            vel_tv = Velocity(shape=(h, w))
            vel_tv.u.fill(0.0) ; vel_tv.v.fill(0.0)
            opt_tv = Optimization(alpha=alpha)
            opt_tv.ForwardBackwardTVL2(grad_engine, vel_tv, nit=nit, tol=tol)
            times["FB TV-L2 (Chambolle)"].append(opt_tv.convergence_time)
            
    finally:
        handler.release()

    # --- ANALYSE ET AFFICHAGE DES RÉSULTATS ---
    print("\n" + "="*45)
    print("BILAN DU BENCHMARK TEMPOREL (Moyenne/frame)")
    print("="*45)
    
    avg_times = {}
    for m in methods:
        avg_t = np.mean(times[m])
        avg_times[m] = avg_t
        print(f"{m:30s} : {avg_t:.4f} secondes")

    # --- TRACÉ DU GRAPHIQUE BAR PLOT ---
    plt.figure(figsize=(8, 6))
    
    # Couleurs ajustées pour 2 barres (Bleu et Vert)
    bars = plt.bar(avg_times.keys(), avg_times.values(), color=['#3498db', '#2ecc71'], width=0.5)
    
    plt.title("Comparaison des temps de calcul moyens par frame", fontsize=14)
    plt.ylabel("Temps moyen (secondes)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajout des valeurs au-dessus des barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f} s", 
                 ha='center', va='bottom', fontweight='bold')

    # Sauvegarde du graphique
    output_plot_path = os.path.join(project_root, "results", "time_benchmark_plot.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    V_PATH = r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet\data\360p_RigidBodyMovement.mp4"
    
    # Lancement sans limite de frames
    run_time_benchmark(V_PATH, alpha=1.0, nit=100, tol=5e-4)