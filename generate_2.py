import os
import cv2
import numpy as np
import uuid
from typing import Tuple


def simulate_random_shapes(
        shape: str,
        output_folder: str,
        sample_size: int,
        length_mean: float = -1,
        length_std: float = -1,
        thickness: int = -1,
        radius_mean: float = -1,
        radius_std: float = -1,
        scale: Tuple[int, int] = (512, 512),
        noise_ratio: float = 0.01,
        border_ratio: Tuple[float, float] = (0.2, 0.8),
        convert_to_rgb: bool = False
) -> None:
    """
    Args:
        shape: the geometric shapes to generate
        output_folder: the folder to save random images
        sample_size: the number of random images to generate
        length_mean: mean length for shape line
        length_std: std of length for shape line
        thickness: thickness for shape line
        radius_mean: mean radius for shape circle
        radius_std: std pf mean radius for shape circle
        scale: the height and width of the image to generate
        noise_ratio: the proportion of noisy points
        border_ratio: where the starting, ending points or circle center are not in
        convert_to_rgb: convert grayscale to rgb channel

    Returns:
        None

    """
    # create the saving folder if not existent
    if not os.path.exists(os.path.join(output_folder)):
        os.makedirs(os.path.join(output_folder))

    # generate images through loop
    for i in range(sample_size):
        img = np.random.binomial(1, noise_ratio, scale).astype(float)

        if shape == "line":
            # Need randomize the starting and ending coordinates for line
            starting_point = np.append(
                np.random.uniform(scale[0] * border_ratio[0], scale[0] * border_ratio[1], 1).astype(int),
                np.random.uniform(scale[0] * border_ratio[0], scale[1] * border_ratio[1], 1).astype(int)
            )
            if length_mean < 0:
                length_mean = scale[0] / 10
            if length_std < 0:
                length_std = scale[0] / 100
            ending_point = tuple(starting_point + length_mean * np.random.normal(0, length_std, 2).astype(int))
            starting_point = tuple(starting_point)
            img_new = cv2.line(img, starting_point, ending_point, color=1, thickness=thickness)
        elif shape == "circle":
            # Need to randomize the center coordinate and the radius value
            center = np.append(
                np.random.uniform(scale[0] * border_ratio[0], scale[0] * border_ratio[1], 1).astype(int),
                np.random.uniform(scale[0] * border_ratio[0], scale[1] * border_ratio[1], 1).astype(int)
            )
            center = tuple(center)
            radius = max(int((np.random.normal(0, radius_std, 1) + radius_mean)[0]), 2)
            img_new = cv2.circle(img, center, radius, 1, -1)
        elif shape == "square":
            # Recommend keeping this structure for future work
            print("The shape specified is wrong or not included yet. No images are generated.")
            return
        elif shape == "triangle":
            # Recommend keeping this structure for future work
            print("The shape specified is wrong or not included yet. No images are generated.")
            return
        elif shape == "noise":
            img_new = img
        else:
            print("The shape specified is wrong or not included yet. No images are generated.")
            return

        if convert_to_rgb:
            # Convert (height, width) to (height, width, 3) with duplications
            img_new = np.stack((img_new, img_new, img_new), axis=2)

        uuid_str = uuid.uuid4().hex
        output_path = os.path.join(output_folder, uuid_str + '.jpg')
        cv2.imwrite(output_path, img_new * 255)


simulate_random_shapes(
        shape = "circle",
        sample_size = 5,
        radius_mean = 10,
        radius_std = 2,
        output_folder="C:/Temp/simulated_images/circles/",
        scale = (256, 256),
        convert_to_rgb=False
)

simulate_random_shapes(
        shape = "line",
        sample_size = 5,
        length_mean = 20,
        length_std = 2,
        thickness = 2,
        output_folder="C:/Temp/simulated_images/lines/",
        scale = (256, 256),
        convert_to_rgb=False
)


simulate_random_shapes(
        shape = "noise",
        sample_size = 5,
        output_folder="C:/Temp/simulated_images/noises/",
        scale = (256, 256)
)

x = cv2.imread("C:/Temp/simulated_images/cricles/cb61ffc2184846aa9f4701732dec8d24.jpg")
