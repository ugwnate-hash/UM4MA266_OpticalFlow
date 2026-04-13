# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
from scipy.ndimage import convolve
import os
import sys

class VideoDataHandler:
    """ 
    Cette classe permet de preprocess (conversion en niveau de gris sur [0,1])
    et obtenir les frames i et i+1 issues de la video choisie.
    """
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path) 
        if not self.cap.isOpened():
            raise ValueError("Erreur lors de l'ouverture du flux vidéo.")
            
    def get_initial_frame(self):
        """Récupère la toute première frame en gris/float."""
        ret, frame = self.cap.read() 
        if not ret:
            return None
        return self._preprocess(frame)

    def _preprocess(self, frame):
        """Conversion standard pour les calculs de gradients."""
        # Passage en gris pour l'hypothèse de constance d'intensité 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        # Conversion en float32 pour la précision des dérivées 
        return gray.astype(np.float32) / 255.0

    def stream_frames(self):
        """Générateur pour lire la vidéo paire par paire."""
        prev_frame = self.get_initial_frame()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            curr_frame = self._preprocess(frame)
            yield prev_frame, curr_frame  # On renvoie le couple (I_t, I_{t+1})
            prev_frame = curr_frame

    def release(self):
        self.cap.release() 

class GradIntensity:
    """
    Cette classe a pour but d'extraire Ex, Ey et Et pour deux frames i et i+1
    issues du stream_frames.
    """
    def __init__(self):
        # Définition des masques 2x2
        # Le facteur 0.25 (1/4) vient de la moyenne des 4 arêtes du cube
        self.kernel_x = np.array([[-1, 1], 
                                  [-1, 1]]) * 0.25
        
        self.kernel_y = np.array([[-1, -1], 
                                  [ 1,  1]]) * 0.25
        
        self.kernel_t = np.array([[ 1,  1], 
                                  [ 1,  1]]) * 0.25

    def compute(self, prev_frame, curr_frame):
        """
        Calcule Ex, Ey et Et à partir de deux images successives.
        prev_frame (I_t) : Image à l'instant k
        curr_frame (I_t+1) : Image à l'instant k+1
        """
        # Ex : Moyenne des gradients x sur les deux frames 
        Ex = convolve(prev_frame, self.kernel_x) + convolve(curr_frame, self.kernel_x)
        
        # Ey : Moyenne des gradients y sur les deux frames 
        Ey = convolve(prev_frame, self.kernel_y) + convolve(curr_frame, self.kernel_y)
        
        # Et : Moyenne des différences temporelles entre les frames 
        Et = convolve(curr_frame, self.kernel_t) - convolve(prev_frame, self.kernel_t)
        
        return Ex, Ey, Et
        

class Velocity:
    """
    Cette classe permet de récupérer les champs des vitesses, 
    le gradient ainsi que le laplacien de u et v.
    """
    def __init__(self, shape):
        # Initialisation des vitesses à zéro (matrices de la taille de l'image)
        self.u = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        
        # Noyau de moyenne locale pondérée
        # Poids : 1/6 pour les voisins directs, 1/12 pour les diagonales 
        self.kernel_avg = np.array([[1/12, 1/6, 1/12],
                                    [1/6,    0, 1/6],
                                    [1/12, 1/6, 1/12]], dtype=np.float32)

    def compute_averages(self):
        """
        Calcule la moyenne locale u_bar et v_bar nécessaire pour 
        l'étape itérative.
        """
        u_bar = convolve(self.u, self.kernel_avg)
        v_bar = convolve(self.v, self.kernel_avg)
        return u_bar, v_bar

    def update(self, new_u, new_v):
        """Met à jour le champ de vitesse actuel."""
        self.u = new_u
        self.v = new_v
    
    def laplacian(self):
        u_bar, v_bar = self.compute_averages()
        return self.u-u_bar, self.v-v_bar
        
        
        
class Optimization:
    """
    Cette classe permet la recherche de u et v pour minimiser l'équation du 
    champ optique ainsi que la contrainte de régularisation.
    """   
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.err_history = []

    def GaussSeidel(self, grad_obj, velocity_obj, nit, tol=1e-6):
        """
        nit : nombre maximum d'itérations.
        tol : seuil de tolérance pour la convergence (epsilon).
        """
        Ex, Ey, Et = grad_obj.Ex, grad_obj.Ey, grad_obj.Et
        alpha_sq = self.alpha**2
        denominator = alpha_sq + Ex**2 + Ey**2

        for i in range(nit):
            # 1. Sauvegarde des anciennes valeurs pour calculer l'écart
            u_old = velocity_obj.u.copy()
            v_old = velocity_obj.v.copy()

            # 2. Calcul des moyennes et mise à jour
            u_bar, v_bar = velocity_obj.compute_averages()
            term_common = (Ex * u_bar + Ey * v_bar + Et) / denominator
            
            u_next = u_bar - Ex * term_common
            v_next = v_bar - Ey * term_common

            # 3. Calcul de la différence (RMSE)
            # On mesure combien le champ a "bougé" pendant cette itération
            diff = np.sqrt(np.mean((u_next - u_old)**2 + (v_next - v_old)**2))
            self.err_history.append(diff)

            # 4. Injection dans l'objet Velocity
            velocity_obj.update(u_next, v_next)

            # 5. Condition d'arrêt
            if diff < tol:
                print(f"Convergence atteinte à l'itération {i} (diff = {diff:.2e})")
                break
            

if __name__ == "__main__":

    video_path = r"C:\Users\ugwna\OneDrive\Cours_3A\Optimisation et science des données\Projet\data\20260304_110902.mp4"
    
    # Configuration des dossiers et noms
    base_name = os.path.basename(video_path)
    file_name = os.path.splitext(base_name)[0]
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{file_name}_scanned.mp4")

    # Initialisation
    handler = VideoDataHandler(video_path)
    total_frames = int(handler.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = handler.cap.get(cv2.CAP_PROP_FPS)
    width = int(handler.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(handler.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    prev_frame = handler.get_initial_frame()
    
    # Initialisation de l'enregistreur
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    velocity = Velocity(shape=prev_frame.shape)
    optimizer = Optimization(alpha=1e-2)
    grad_engine = GradIntensity()
    
    hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255 
    
    count = 0
    try:
        print(f"Démarrage du scan : {file_name}")
        print(f"Total de frames à traiter : {total_frames}")
        
        while True:
            ret, frame_color = handler.cap.read()
            if not ret:
                break
            
            count += 1
            curr_frame = handler._preprocess(frame_color)
            
            # Calcul des gradients et Optimisation
            grad_engine.Ex, grad_engine.Ey, grad_engine.Et = grad_engine.compute(prev_frame, curr_frame)
            optimizer.GaussSeidel(grad_engine, velocity, nit=100, tol=1e-5)
            
            # Visualisation HSV
            mag, ang = cv2.cartToPolar(velocity.u, velocity.v)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Sauvegarde et affichage
            writer.write(rgb_flow)
            cv2.imshow('Flot Optique en cours...', rgb_flow)
            
            # Affichage de la progression dans la console (sur une seule ligne)
            percent = (count / total_frames) * 100
            sys.stdout.write(f"\rProgression : [{count}/{total_frames}] {percent:.1f}%")
            sys.stdout.flush()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nInterruption par l'utilisateur.")
                break
                
            prev_frame = curr_frame
        
        print(f"\nVidéo sauvegardée dans : {output_path}")

    finally:
        handler.release()
        writer.release()
        cv2.destroyAllWindows()
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        