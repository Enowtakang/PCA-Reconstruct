import glob
from matplotlib import image as img
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


path = "healthy/*.*"
count = 0

for file in glob.glob(path):
    image = img.imread(file)
    # print(image.shape)
    image_reshaped = np.reshape(image, (4000, 6000 * 3))
    # print(image_reshaped.shape)
    pca = PCA(600).fit(image_reshaped)
    image_transformed = pca.transform(image_reshaped)
    # print(image_transformed.shape)
    image_inverse_transformed = pca.inverse_transform(
        image_transformed)

    image_reconstructing = np.reshape(
        image_inverse_transformed, (4000, 6000, 3))

    count += 1

    plt.axis('off')
    plt.imshow(image_reconstructing.astype('uint8'))
    plt.savefig(
        f"recon_healthy/recon_diz_{count}.JPG",
        bbox_inches='tight',
        pad_inches=0.0)


"""
REFERENCES:
    How to batch process multiple images in python
    https://www.youtube.com/watch?v=QxzxLVzNfbI
"""
