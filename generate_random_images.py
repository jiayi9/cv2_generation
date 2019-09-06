
import cv2
import numpy as np
from matplotlib import pyplot as plt
import uuid

def generate_random_images(mode, output_folder, N, 
                           length_mean = None, length_std= None, thickness = 2, 
                           radius_mean = None, radius_std = None, 
                           scale = (200,200), ratio = 0.01, border_ratio = (0.2, 0.8)):
    for i in range(N):
        img = np.random.binomial(1, ratio, scale).astype(float)
        
        if mode == "line":
            starting_point = np.append(
                            np.random.uniform(scale[0]*border_ratio[0], scale[0]*border_ratio[1], 1).astype(int),
                            np.random.uniform(scale[0]*border_ratio[0], scale[1]*border_ratio[1], 1).astype(int)
                            )
            if length_mean == None:
                length_mean = scale[0]/10
            if length_std == None:
                length_std = scale[0]/100
            ending_point = tuple(starting_point  + length_mean*np.random.normal(0, length_std, 2).astype(int))
            starting_point = tuple(starting_point)
            new = cv2.line(img, starting_point, ending_point, color = 1, thickness = thickness)
        if mode == "circle":
            center = np.append(
                            np.random.uniform(scale[0]*border_ratio[0], scale[0]*border_ratio[1], 1).astype(int),
                            np.random.uniform(scale[0]*border_ratio[0], scale[1]*border_ratio[1], 1).astype(int)
                            )
            center = tuple(center)
            RADIUS = max(int((np.random.normal(0, radius_std, 1) + radius_mean)[0]),2)
            new = cv2.circle(img, center, RADIUS, 1, -1)
        if mode == "square":
            print("Not done yet!")
            break
        if mode == "triangle":
            print("Not done yet!")
            break
        if mode == "pass":
            new = img
        uuid_str = uuid.uuid4().hex
        cv2.imwrite(output_folder + uuid_str + '.jpg', new*255)




generate_random_images(
        mode = "line",
        output_folder="C:/LV_CHAO_IMAGE/simulation_data/line_256/",
        N = 1000,
        length_mean = 20,
        length_std = 5,
        thickness = 3,
        scale = (256,256)
)

generate_random_images(
        mode = "circle",
        N = 1000,
        radius_mean = 15,
        radius_std = 4,
        output_folder="C:/LV_CHAO_IMAGE/simulation_data/circle_256/",
        scale = (256,256)
)

generate_random_images(
        mode = "pass",
        N = 1000,
        output_folder="C:/LV_CHAO_IMAGE/simulation_data/pass_256/",
        scale = (256,256)
)

