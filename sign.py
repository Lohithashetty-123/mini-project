import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def match(path1, path2):
    # Read the images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Resize images for comparison
    gray1 = cv2.resize(gray1, (300, 300))
    gray2 = cv2.resize(gray2, (300, 300))

    # Apply binary thresholding
    _, thresh1 = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(gray2, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a black background
    contour_img1 = np.zeros_like(gray1)
    contour_img2 = np.zeros_like(gray2)
    cv2.drawContours(contour_img1, contours1, -1, (255), thickness=cv2.FILLED)
    cv2.drawContours(contour_img2, contours2, -1, (255), thickness=cv2.FILLED)

    # Compute similarity
    similarity_value = "{:.2f}".format(ssim(contour_img1, contour_img2) * 100)

    # Display both images
    cv2.imshow("Contours Image One", contour_img1)
    cv2.imshow("Contours Image Two", contour_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return float(similarity_value)

# Example usage
# ans = match("path_to_image1.png", "path_to_image2.png")
# print(ans)
