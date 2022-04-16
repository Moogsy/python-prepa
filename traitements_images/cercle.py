import numpy as np
from matplotlib import pyplot as plt

img = np.zeros((1024, 1024, 3), dtype=np.uint8)
img[:, :, :] = 255

a, b, _ = img.shape
x, y = a / 2, b / 2

red = np.array([255, 0, 0])

for i in range(a):
    for j in range(b):
        if (x - i) ** 2 + (x - j) ** 2 < 5000:
            img[i, j] = red

fig, ax = plt.subplots()
ax.imshow(img)
plt.show()

