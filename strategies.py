import cv2
import numpy as np 

def create_window(win_name, win_size, win_position):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, *win_size)
    cv2.moveWindow(win_name, *win_position)

def build_matrix(points):
    nb_points, _ = points.shape 
    response = np.zeros((nb_points, nb_points))
    for i_ptr in range(nb_points):
        for j_ptr in range(i_ptr, nb_points):
            fst = points[i_ptr]
            snd = points[j_ptr]
            dst = np.sqrt(np.sum((snd - fst) ** 2) + 1e-8)
            response[i_ptr, j_ptr] = dst 
            response[j_ptr, i_ptr] = dst 
    
    return response 
            

def draw_landmarks(mp_drawing, mp_drawing_styles, mp_hands, image, hand_landmarks):
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
        