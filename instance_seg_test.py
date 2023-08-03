import cv2
import numpy as np

# Read the image
image = cv2.imread('instance_seg_test.png')

# OpenCV reads colors as BGR; the red channel is the last one
B, G, R = cv2.split(image)

# Create a binary mask where red == 10
mask = np.where(R == 10, 255, 0)

# The mask is a float array, but images are usually 8-bit, so convert it
mask = mask.astype(np.uint8)

# Mask the G and B channels
G = np.where(mask == 255, G, 0)
B = np.where(mask == 255, B, 0)

# Stack G and B to create a 2D image for unique pairs
GB = np.dstack((G, B))

# Find unique pairs in the GB image
unique_pairs = np.unique(GB.reshape(-1, GB.shape[-1]), axis=0)

# Create a copy of the original image to draw bounding boxes on
image_with_boxes = image.copy()

# For each unique pair, draw a bounding box on the image
for pair in unique_pairs:
    # print("in the loop")
    # Check if pair is not [0,0] (which is not a vehicle)
    if np.any(pair > 0):
        # Find the pixels that have this G-B pair
        pixels = np.where(np.all(GB == pair, axis=-1))
        
        # Get the bounding box coordinates
        x_min, x_max = np.min(pixels[1]), np.max(pixels[1])
        y_min, y_max = np.min(pixels[0]), np.max(pixels[0])
        
        # Draw the bounding box
        print("drawing a box")
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Save the image with bounding boxes
cv2.imwrite('image_with_boxes.png', image_with_boxes)

# Display the image with bounding boxes
cv2.imshow('Image with boxes', image_with_boxes)
