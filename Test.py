import cv2
import numpy as np

from utils.segmentation import Segmentation


# ==========================================
# LOAD IMAGE
# ==========================================

IMAGE_PATH = r"cnn_test.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Could not load image")
    exit()

print("Image loaded:", image.shape)


# ==========================================
# LOAD SEGFORMER
# ==========================================

segmenter = Segmentation()


# ==========================================
# SEGMENT
# ==========================================

road_roi, road_mask, coverage = segmenter.process(
    image
)


# ==========================================
# RESULTS
# ==========================================

print(
    f"Coverage: {coverage:.2f}%"
)

print(
    "Road pixels:",
    cv2.countNonZero(road_mask)
)


# ==========================================
# DISPLAY
# ==========================================

cv2.imshow(
    "Original",
    image
)

cv2.imshow(
    "Road Mask",
    road_mask
)

cv2.imshow(
    "Road ROI",
    road_roi
)

cv2.waitKey(0)

cv2.destroyAllWindows()