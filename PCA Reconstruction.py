from matplotlib import image as img
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


"""
1. Read Image
"""
path = "path/to/file"
image = img.imread(path)
# print(image.shape)

"""
2. Reshape image to PCA understandable format
"""
image_reshaped = np.reshape(image, (4000, 6000*3))
# print(image_reshaped.shape)

"""
3. PCA with n components
            (n = 600)
"""
pca = PCA(600).fit(image_reshaped)
image_transformed = pca.transform(image_reshaped)
# print(image_transformed.shape)


"""
4. Inverse transform to recreate original dimension
"""
image_inverse_transformed = pca.inverse_transform(
    image_transformed)

image_reconstructing = np.reshape(
    image_inverse_transformed, (4000, 6000, 3))

"""
5. Display compressed image
"""
plt.axis('off')
plt.imshow(image_reconstructing.astype('uint8'))
plt.savefig(
    'hello.jpg',
    bbox_inches='tight',
    pad_inches=0.0)
# plt.show()
