# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:51:34 2026

@author: ugwna
"""

import os
import cv2

def extract_comparison_frames(path_method1, path_method2, target_frames, output_base_dir, alpha_val):
    """
    Extrait les mêmes frames de deux vidéos différentes pour faire une comparaison visuelle.
    """
    # Vérification de l'existence des fichiers
    if not os.path.exists(path_method1):
        print(f"Erreur : Vidéo Méthode 1 introuvable -> {path_method1}")
        return
    if not os.path.exists(path_method2):
        print(f"Erreur : Vidéo Méthode 2 introuvable -> {path_method2}")
        return

    cap1 = cv2.VideoCapture(path_method1)
    cap2 = cv2.VideoCapture(path_method2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Erreur d'ouverture d'une des vidéos.")
        return

    print(f"Extraction pour Alpha = {alpha_val}...")

    for frame_idx in target_frames:
        # Placement des deux curseurs de lecture sur la même frame
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2:
            # Création d'un dossier dédié pour cette frame de comparaison
            frame_dir = os.path.join(output_base_dir, f"Comparaison_Methodes_frame_{frame_idx:04d}")
            os.makedirs(frame_dir, exist_ok=True)
            
            # Sauvegarde Horn-Schunck (Méthode 1)
            name1 = f"HS_frame_{frame_idx:04d}_alpha_{alpha_val}.png"
            path_out1 = os.path.join(frame_dir, name1)
            cv2.imwrite(path_out1, frame1)
            
            # Sauvegarde TV-L2 (Méthode 2)
            name2 = f"TVL2_frame_{frame_idx:04d}_alpha_{alpha_val}.png"
            path_out2 = os.path.join(frame_dir, name2)
            cv2.imwrite(path_out2, frame2)
            
            print(f" -> Frame {frame_idx:04d} extraite avec succès (HS et TVL2).")
        else:
            print(f"Attention : Frame {frame_idx} introuvable dans l'une des vidéos.")

    cap1.release()
    cap2.release()
    print("-" * 40)


if __name__ == "__main__":
    VIDEO_NAME = "360p_RigidBodyMovement"  
    
    # Les frames que l'on a analysées précédemment
    TARGET_FRAMES = [40, 93] 
    
    # L'Alpha de référence pour comparer à armes égales
    ALPHA = 1.0 
    EXTENSION = ".avi" # Vérifie que c'est bien .avi et non .mp4
    
    # Chemins de base de ton projet
    DIR_METHOD1 = "results"          # Dossier contenant Gauss-Seidel (Horn-Schunck)
    DIR_METHOD2 = "results_TVL1"     # Dossier contenant ton TV-L2
    LATEX_IMG_DIR = "latex_images_export"

    # Construction des chemins exacts vers les vidéos HSV
    video_hs = os.path.join(DIR_METHOD1, VIDEO_NAME, f"alpha_{ALPHA}", f"{VIDEO_NAME}_hsv{EXTENSION}")
    video_tvl2 = os.path.join(DIR_METHOD2, VIDEO_NAME, f"alpha_{ALPHA}", f"{VIDEO_NAME}_hsv{EXTENSION}")
    
    print(f"Comparaison de :")
    print(f"1. {video_hs}")
    print(f"2. {video_tvl2}\n")

    extract_comparison_frames(video_hs, video_tvl2, TARGET_FRAMES, LATEX_IMG_DIR, ALPHA)
    
    print("\nExtraction comparative terminée ! Tes images sont dans le dossier d'export.")