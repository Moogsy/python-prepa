import itertools

import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image

im = np.zeros((256, 256, 3), dtype=np.uint8)

line = np.linspace(0, 255, 256, dtype=np.uint8)

red = np.meshgrid(line, line)[0]


blue = red.transpose()
green = np.zeros((256, 256))




im[:, :, 0] = red 
im[:, :, 1] = green 
im[:, :, 2] = blue 

fig, ax = plt.subplots()

ax.imshow(im)
ax.imshow(im)
plt.show()
