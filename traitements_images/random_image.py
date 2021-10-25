from random import randint 

import numpy as np
from matplotlib import pyplot as plt


img = np.zeros((256, 256, 3), dtype=np.uint8)

a, b, _ = img.shape
x, y = a / 2, b / 2

for i in range(a):
    for j in range(b):
        img[i, j] = np.array([randint(0, 255), randint(0, 255), randint(0, 255)])

fig, ax = plt.subplots()
ax.imshow(img)
plt.show()

