import cv2
import numpy as np
from skimage.segmentation import felzenszwalb
import matplotlib.pyplot as plt
# Load image
img = cv2.imread('Project2/0001_3.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
gray = cv2.GaussianBlur(gray,(5,5),0)


# Perform Felzenszwalb segmentation
segments = felzenszwalb(gray, scale=500, sigma=0.8, min_size=80)

# Create markers image
markers = np.zeros(img.shape[:2], dtype=np.int32)

# Assign label to each segment
for i, segment in enumerate(np.unique(segments)):
    markers[segments == segment] = i + 1

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(markers, cmap="tab20b")
ax.axis('off')
plt.show()

# Perform Watershed
#cv2.watershed(img, markers)
# Apply the Watershed algorithm to segment the image
markers = cv2.watershed(gray, markers)

# Draw segmentation boundaries
mask = np.zeros(img.shape[:2], dtype="uint8")
for index in np.unique(markers)[1:]:
    mask[markers == index] = 255
result = cv2.bitwise_and(img, img, mask= mask)
#show the images

cv2.imshow("Segmentation", result)
cv2.waitKey(0)
cv2.destroyAllWindows()