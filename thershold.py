import cv2

# Load the image
img = cv2.imread('Scanner02.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Crop seeds and save as individual images
for i, cnt in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cnt)
    roi = img[y:y+h, x:x+w]
    cv2.imwrite('seed_{}.jpg'.format(i), roi)
