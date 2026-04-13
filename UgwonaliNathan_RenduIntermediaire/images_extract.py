import os
import cv2

def extract_specific_frames(video_path, target_frames, output_base_dir, alpha_val, prefix=""):
    """Extrait et sauvegarde des frames spécifiques d'une vidéo."""
    if not os.path.exists(video_path):
        print(f"Fichier introuvable : {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur d'ouverture : {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    clean_name = video_name.replace("_hsv", "").replace("_vector", "")

    for frame_idx in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame_dir = os.path.join(output_base_dir, f"{clean_name}_frame_{frame_idx:04d}")
            os.makedirs(frame_dir, exist_ok=True)
            
            file_name = f"{prefix}_{clean_name}_frame_{frame_idx:04d}_alpha_{alpha_val}.png"
            out_path = os.path.join(frame_dir, file_name)
            
            cv2.imwrite(out_path, frame)
            print(f"Export réussi -> {out_path}")
        else:
            print(f"Attention : Frame {frame_idx} introuvable dans {clean_name} (Alpha {alpha_val})")

    cap.release()

if __name__ == "__main__":
    VIDEO_NAME = "360p_F1"  
    PERCENTAGES = [0.10, 0.50, 0.90]
    
    RESULTS_DIR = os.path.join("results", VIDEO_NAME)
    LATEX_IMG_DIR = "latex_images_export"
    alphas = [0.01, 1.0, 100.0]

    first_video_path = os.path.join(RESULTS_DIR, f"alpha_{alphas[0]}", f"{VIDEO_NAME}_hsv.mp4")
    
    cap_temp = cv2.VideoCapture(first_video_path)
    if not cap_temp.isOpened():
        print(f"Erreur critique: Impossible d'ouvrir {first_video_path}")
        exit()
        
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()

    frames_to_extract = [int(total_frames * p) for p in PERCENTAGES]
    
    print(f"Vidéo : {total_frames} frames. Cibles : {frames_to_extract}\n")

    for alpha in alphas:
        alpha_folder = os.path.join(RESULTS_DIR, f"alpha_{alpha}")
        
        video_hsv = os.path.join(alpha_folder, f"{VIDEO_NAME}_hsv.mp4")
        video_vec = os.path.join(alpha_folder, f"{VIDEO_NAME}_vector.mp4")
        
        extract_specific_frames(video_hsv, frames_to_extract, LATEX_IMG_DIR, alpha, prefix="HSV")
        extract_specific_frames(video_vec, frames_to_extract, LATEX_IMG_DIR, alpha, prefix="VEC")

    print("\nExtraction terminée.")