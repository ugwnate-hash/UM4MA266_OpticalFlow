# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:34:38 2026

@author: ugwna
"""
import numpy as np
import time  # Ajout du module pour le temps
from tqdm import tqdm

class Optimization:
    """
    Cette classe permet la recherche de u et v pour minimiser l'équation du 
    champ optique ainsi que la contrainte de régularisation.
    """   
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.err_history = []
        self.convergence_time = 0.0  # Pour sauvegarder le temps total de calcul

    def GaussSeidel(self, grad_obj, velocity_obj, nit, tol=1e-6):
        """
        nit : nombre maximum d'itérations.
        tol : seuil de tolérance pour la convergence (epsilon).
        """
        Ex, Ey, Et = grad_obj.Ex, grad_obj.Ey, grad_obj.Et
        alpha_sq = self.alpha**2
        denominator = alpha_sq + Ex**2 + Ey**2

        pbar = tqdm(range(nit), desc="Calcul du flot", unit="iter")
        
        # DÉMARRAGE DU CHRONOMÈTRE
        start_time = time.perf_counter()

        for i in pbar:
            # 1. Sauvegarde des anciennes valeurs pour calculer l'écart
            u_old = velocity_obj.u.copy()
            v_old = velocity_obj.v.copy()

            # 2. Calcul des moyennes et mise à jour
            u_bar, v_bar = velocity_obj.compute_averages()
            term_common = (Ex * u_bar + Ey * v_bar + Et) / denominator
            
            u_next = u_bar - Ex * term_common
            v_next = v_bar - Ey * term_common

            # 3. Calcul de la différence (RMSE)
            diff = float(np.sqrt(np.mean((u_next - u_old)**2 + (v_next - v_old)**2)))
            self.err_history.append(diff)

            pbar.set_postfix({"diff": f"{diff:.2e}"})

            # 4. Injection dans l'objet Velocity
            velocity_obj.update(u_next, v_next)

            # 5. Condition d'arrêt
            if diff < tol:
                # ARRÊT DU CHRONOMÈTRE (Convergence atteinte)
                end_time = time.perf_counter()
                self.convergence_time = end_time - start_time
                
                pbar.set_description("Convergence atteinte")
                pbar.close()
                print(f"\n[Succès] Tolérance respectée en {i} itérations.")
                print(f"Temps de calcul : {self.convergence_time:.4f} secondes.\n")
                return # On sort de la fonction

        # ARRÊT DU CHRONOMÈTRE (Si le max d'itérations est atteint sans converger)
        end_time = time.perf_counter()
        self.convergence_time = end_time - start_time
        pbar.close()
        print(f"\n[Terminé] Limite de {nit} itérations atteinte sans convergence parfaite.")
        print(f"Temps de calcul : {self.convergence_time:.4f} secondes.\n")
        
    def GradientDescent(self, grad_obj, velocity_obj, nit, lr=0.001, tol=1e-6):
        """
        Résolution par descente de gradient standard à pas constant.
        lr : pas d'apprentissage (learning rate). Attention, s'il est trop grand, l'algo diverge.
        """
        import time
        from tqdm import tqdm
        
        Ex, Ey, Et = grad_obj.Ex, grad_obj.Ey, grad_obj.Et
        alpha_sq = self.alpha**2

        pbar = tqdm(range(nit), desc="Calcul du flot (Gradient Descent)", unit="iter")
        start_time = time.perf_counter()

        for i in pbar:
            # 1. Sauvegarde des valeurs courantes
            u = velocity_obj.u.copy()
            v = velocity_obj.v.copy()

            # 2. Calcul des moyennes locales (Laplacien discret)
            u_bar, v_bar = velocity_obj.compute_averages()

            # 3. Calcul du terme d'attache aux données commun
            data_term = Ex * u + Ey * v + Et

            # 4. Calcul des gradients
            grad_u = Ex * data_term - alpha_sq * (u_bar - u)
            grad_v = Ey * data_term - alpha_sq * (v_bar - v)

            # 5. Mise à jour par descente de gradient
            u_next = u - lr * grad_u
            v_next = v - lr * grad_v

            # 6. Évaluation de la convergence (RMSE du déplacement)
            diff = float(np.sqrt(np.mean((u_next - u)**2 + (v_next - v)**2)))
            self.err_history.append(diff)
            pbar.set_postfix({"diff": f"{diff:.2e}"})

            # 7. Injection dans l'objet Velocity
            velocity_obj.update(u_next, v_next)

            # 8. Condition d'arrêt
            if diff < tol:
                self.convergence_time = time.perf_counter() - start_time
                pbar.set_description("Convergence atteinte")
                pbar.close()
                print(f"\n[Succès GD] Tolérance respectée en {i} itérations.")
                print(f"Temps de calcul : {self.convergence_time:.4f} secondes.\n")
                return

        # Si limite d'itérations atteinte
        self.convergence_time = time.perf_counter() - start_time
        pbar.close()
        print(f"\n[Terminé GD] Limite de {nit} itérations atteinte.")
        print(f"Temps de calcul : {self.convergence_time:.4f} secondes.\n")
        