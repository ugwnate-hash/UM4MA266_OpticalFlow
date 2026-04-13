# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:32:52 2026

@author: ugwna
"""

import numpy as np
from scipy.ndimage import convolve


class GradIntensity:
    """
    Cette classe a pour but d'extraire Ex, Ey et Et pour deux frames i et i+1
    issues du stream_frames.
    """
    def __init__(self):
        # Définition des masques 2x2
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