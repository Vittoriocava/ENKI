import numpy as np
from scipy.ndimage import zoom

matrix = np.random.rand(4, 4)
# To go from 4x4 to 512x512, the zoom factor is 512/4 = 128.
# Will the resultant shape be exactly 512x512?
zoomed = zoom(matrix, 128, order=3)
print(matrix.shape, zoomed.shape)
