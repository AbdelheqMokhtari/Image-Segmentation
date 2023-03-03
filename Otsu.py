import cv2

# Load the image
img = cv2.imread('test.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)
cv2.waitKey(0)

# Apply thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Apply connected component analysis
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

# Loop over each seed and crop it
for i in range(1, n_labels):
    # Get the bounding box coordinates
    x, y, w, h, area = stats[i]

    # Crop the region from the original image
    if 10 < area:
        seed = img[y:y+h, x:x+w]
        print(i)
        # Save the cropped seed image
        cv2.imwrite(f'data/seed_{i}.png', seed)
