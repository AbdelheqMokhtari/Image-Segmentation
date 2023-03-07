import cv2


class SeedAligner:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def preprocess(self):
        # convert image to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # apply Gaussian blur to smooth the image
        self.blur = cv2.GaussianBlur(self.gray, (5, 5), 0)

        # perform Canny edge detection
        self.edges = cv2.Canny(self.blur, 100, 200)

    def detect_contours(self):
        # find contours in the edge map
        contours, hierarchy = cv2.findContours(self.edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for contour in contours:
            # calculate the orientation angle using PCA
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)

            # rotate the contour by the angle
            M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
            rotated_contour = cv2.warpAffine(contour, M, self.image.shape[:2])

            # display the rotated contour
            cv2.imshow("Rotated contour", rotated_contour)
            cv2.waitKey(0)


if __name__ == "__main__":
    aligner = SeedAligner("seed_image.jpg")
    aligner.preprocess()
    aligner.detect_contours()
