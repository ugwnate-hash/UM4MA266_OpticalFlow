# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:33:54 2026

@author: ugwna
"""

import numpy as np
from scipy.ndimage import convolve


class Velocity:
    """
    Cette classe permet de récupérer les champs des vitesses, 
    le gradient ainsi que le laplacien de u et v.
    """
    def __init__(self, shape):
        # Initialisation des vitesses à zéro (matrices de la taille de l'image)
        self.u = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        
        # Noyau de moyenne locale pondérée (Figure 3)
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