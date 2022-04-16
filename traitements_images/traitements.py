import numpy as np 
from matplotlib import pyplot as plt 
from PIL import Image

def gris(img: np.ndarray) -> np.ndarray:
    n, p, *_ = img.shape
    mat_gris = np.zeros((n, p), dtype=float)

    for i in range(n):
        for j in range(p):

            r, g, b, *_ = img[i, j]

            mat_gris[i, j] = r / 3 + g / 3 + b / 3

    return mat_gris

def negatif(img: np.ndarray) -> np.ndarray:
    return 255 - img

def floutage(img: np.ndarray, n: int = 5) -> np.ndarray: 
    M = img.copy()

    for _ in range(n):
        M[1:-1, 1:-1, :] = (
            + 0.25 * M[2:, 1:-1, :] 
            + 0.25 * M[:-2, 1:-1, :]
            + 0.25 * M[1:-1, :-2, :]
            + 0.25 * M[1:-1, 2:, :]
        )

    return M

def floutage2(img: np.ndarray, n: int = 5) -> np.ndarray:
    M = img.copy()


    p, q, *_ = M.shape

    for _ in range(n): 
        for i in range(1, p - 1):
            for j in range(1, q - 1):
                M[i, j] = 0.25 * M[i + 1, j] + 0.25 * M[i - 1, j] + 0.25 * M[i, j + 1] + 0.25 * M[i, j -1]

    return M

def contours(img: np.ndarray) -> np.ndarray:
    grey = gris(img).astype(int)

    n, p = grey.shape
    out = grey.copy()


    for i in range(1, n-1):
        for j in range(1, p-1):
            out[i, j] = abs(grey[i, j+1] - grey[i, j-1]) / 2 + abs(grey[i+1, j] - grey[i-1, j]) / 2

    return out

def test():
    img = Image.open("cat.png")
    mat_img = np.asarray(img)


    img_gris = gris(mat_img)
    img_negatif = negatif(gris(mat_img))
    img_floutee = floutage(mat_img)
    img_contours = contours(mat_img)
    img_contours_neg = negatif(img_contours)


    _, ax = plt.subplots(nrows=2, ncols=3)
    ax[0, 0].imshow(mat_img)
    ax[0, 1].imshow(img_gris, cmap="gray")
    ax[1, 0].imshow(img_negatif, cmap="gray")
    ax[1, 1].imshow(img_floutee, cmap="gray")
    ax[0, 2].imshow(img_contours, cmap="gray")
    ax[1, 2].imshow(img_contours_neg, cmap="gray")

    plt.show()






test()
