import os
import time
import cv2
import matplotlib.pyplot as plt
import sys

chemin_dll_conda = r"C:\Users\ugwna\anaconda3\envs\spyenv\Library\bin"
if hasattr(os, 'add_dll_directory') and os.path.exists(chemin_dll_conda):
    os.add_dll_directory(chemin_dll_conda)
sys.path.append(r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet")

from core.video_io import VideoDataHandler
from core.gradients import GradIntensity
from core.velocity import Velocity
from core.optimizer import Optimization

def run_full_benchmark(video_path, alpha_val, nit=50, tol=1e-4, lr_list=[1]):
    print(f"\n--- BENCHMARK COMPLET (Alpha={alpha_val}, Max Iter={nit}) ---")
    print(f"Learning rates testés : {lr_list}")
    
    handler = VideoDataHandler(video_path)
    grad_engine = GradIntensity()
    
    total_frames = int(handler.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    total_time_jacobi = 0.0
    time_gd_dict = {lr: 0.0 for lr in lr_list}
    frames_processed = 0

    try:
        for p_frame, c_frame, _ in handler.stream_frames():
            frames_processed += 1
            print(f"Traitement frame {frames_processed}/{total_frames-1}...")
            
            grad_engine.Ex, grad_engine.Ey, grad_engine.Et = grad_engine.compute(p_frame, c_frame)

            # 1. Mesure Jacobi (Gauss-Seidel) - Référence
            vel_jacobi = Velocity(shape=p_frame.shape)
            opt_jacobi = Optimization(alpha=alpha_val)
            
            t0 = time.perf_counter()
            opt_jacobi.GaussSeidel(grad_engine, vel_jacobi, nit=nit, tol=tol)
            total_time_jacobi += (time.perf_counter() - t0)

            # 2. Mesures Descente de Gradient pour chaque lr
            for lr in lr_list:
                vel_gd = Velocity(shape=p_frame.shape)
                opt_gd = Optimization(alpha=alpha_val)
                
                t0 = time.perf_counter()
                opt_gd.GradientDescent(grad_engine, vel_gd, nit=nit, tol=tol, lr=lr)
                time_gd_dict[lr] += (time.perf_counter() - t0)

    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.")
    finally:
        handler.release()
        
    if frames_processed == 0:
        return None, None, 0

    return total_time_jacobi, time_gd_dict, frames_processed

def plot_benchmark(time_jacobi, time_gd_dict, frames_processed):
    if time_jacobi is None or not time_gd_dict:
        return
        
    methods = ['Gauss-Seidel'] + [f'GD (lr={lr})' for lr in time_gd_dict.keys()]
    times = [time_jacobi] + list(time_gd_dict.values())

    plt.figure(figsize=(10, 6))
    # Attribution d'une couleur distincte pour Gauss-Seidel, et une même teinte pour les GD
    colors = ['#4C72B0'] + ['#DD8452'] * len(time_gd_dict)
    bars = plt.bar(methods, times, color=colors)
    
    for bar, t_total in zip(bars, times):
        yval = bar.get_height()
        fps = frames_processed / t_total if t_total > 0 else 0
        texte_barre = f"{t_total:.1f} s\n({fps:.2f} fps)"
        
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, texte_barre, 
                 va='bottom', ha='center', fontsize=11, fontweight='bold')

    plt.title(f"Sensibilité au pas d'apprentissage ({frames_processed} images)", fontsize=14, pad=20)
    plt.ylabel("Temps cumulé (Secondes)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.ylim(0, max(times) * 1.15)
    
    plt.savefig("benchmark_lr_study.png", dpi=300, bbox_inches='tight')
    print("\nGraphique sauvegardé sous 'benchmark_lr_study.png'")
    
    plt.show()

if __name__ == "__main__":
    V_PATH = r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet\data\360p_RigidBodyMovement.mp4"
    ALPHA = 1
    MAX_ITER = 100 
    TOLERANCE = 1e-4
    LR_LIST = [1.0, 0.1, 0.01]
    
    t_jac, t_gd_dict, n_frames = run_full_benchmark(V_PATH, ALPHA, nit=MAX_ITER, tol=TOLERANCE, lr_list=LR_LIST)
    
    plot_benchmark(t_jac, t_gd_dict, n_frames)