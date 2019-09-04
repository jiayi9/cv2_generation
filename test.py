
import cv2
import numpy as np
from matplotlib import pyplot as plt


img = np.random.random((200,200))

img = np.random.binomial(1, 0.005, (100,100))*255


plt.imshow(img, cmap = "gray")

starting_point = np.random.uniform(100, 100, 2).astype(int)

ending_point = starting_point  + np.random.normal(15, 2, 2).astype(int)

print(starting_point, ending_point)

starting_point = tuple(starting_point.tolist())

ending_point = tuple(ending_point.tolist())

plt.imshow(img)

#new = cv2.line(img, starting_point, ending_point, color = (255, 255, 255), thickness = 2, lineType = 8, shift = 0)
new = cv2.line(img, starting_point, ending_point, color = (255, 255, 255))

# plt.imshow(new, cmap = "gray")

plt.imshow(new)
