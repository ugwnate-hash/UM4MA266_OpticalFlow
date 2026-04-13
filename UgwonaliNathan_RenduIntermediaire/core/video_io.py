# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:31:09 2026

@author: ugwna
"""

import numpy as np
import cv2

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
        """Générateur renvoyant (prev_gray, curr_gray, curr_raw_color)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret: return
        
        prev_gray = self._preprocess(frame)
        
        while True:
            ret, frame = self.cap.read() # frame est en couleur (BGR)
            if not ret: break
            
            curr_gray = self._preprocess(frame)
            yield prev_gray, curr_gray, frame # On renvoie aussi l'image couleur originale
            prev_gray = curr_gray

    

    def release(self):
        self.cap.release() 