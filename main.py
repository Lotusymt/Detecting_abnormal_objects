import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
# define gaussian blur kernel size to remove minor distictions
k_size = 11
def preprocess_image(img):
    # Resize the image

    #Apply histogram equalization to improve contrast
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Apply Gaussian blur to denoise the image
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # blur it futher might be able to remove minor distictions?
    img = cv2.GaussianBlur(img, (k_size, k_size), 0)



    return img
# def calc_interclass_variance(image, threshold):
#     below = image[image <= threshold]
#     above = image[image > threshold]
#     return (np.var(below) + np.var(above)) / 2
#
# def criterion_improves(image, threshold):
#     prev_criterion = 0
#     curr_criterion = calc_interclass_variance(image, threshold)
#     if curr_criterion > prev_criterion:
#         prev_criterion = curr_criterion
#         return True
#     return False

def abnoral_detection(normal_img,abnormal_img):
    #resize abnormal image
    abnormal_img = cv2.resize(abnormal_img, (640, 480))
    normal_img = cv2.resize(normal_img, (640, 480))
    org_gray = cv2.cvtColor(abnormal_img, cv2.COLOR_BGR2GRAY)
    # Convert images to grayscale
    normal_img_ = preprocess_image(normal_img)
    abnormal_img_ = preprocess_image(abnormal_img)


    # show the result
    # cv2.imshow('normal', normal_img_)
    cv2.imshow('abnormal', abnormal_img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    normal_gray = cv2.cvtColor(normal_img_, cv2.COLOR_BGR2GRAY)
    abnormal_gray = cv2.cvtColor(abnormal_img_, cv2.COLOR_BGR2GRAY)


    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Set the scale range
    sift.setSigma(60)  # Set the sigma value to control the scale range

    # Detect keypoints and compute descriptors for both images
    kp1, des1 = sift.detectAndCompute(normal_gray, None)
    kp2, des2 = sift.detectAndCompute(abnormal_gray, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors of normal and abnormal image keypoints
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    #matched_kp1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    #matched_kp2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # Extract the matched keypoints
    matched_kp1 = [kp1[m.queryIdx] for m in matches]
    matched_kp2 = [kp2[m.trainIdx] for m in matches]

    # Find the unmatched keypoints
    unmatched_kp1 = [kp for kp in kp1 if kp not in matched_kp1]
    unmatched_kp2 = [kp for kp in kp2 if kp not in matched_kp2]
    # # Perform Felzenszwalb segmentation
    # segments = felzenszwalb(abnormal_gray, scale=300, sigma=0.5, min_size=30)
    #
    # # Create markers image
    # markers = np.zeros(abnormal_gray.shape[:2], dtype=np.int32)
    #
    # # Assign label to each segment
    # for i, segment in enumerate(np.unique(segments)):
    #     markers[segments == segment] = i + 1

    #plot the histogram of abnormal gray
    plt.hist(org_gray.ravel(), 256, [0, 256])
    plt.show()

    # # Calculate gradient
    # sobel = cv2.Sobel(org_gray, cv2.CV_64F, 1, 1, ksize=5)
    #
    # # Apply gradient threshold
    # _, edge_map = cv2.threshold(sobel, 0.05, 1, cv2.THRESH_BINARY)
    #
    # # Invert edge map
    # #weight_map = 1 - edge_map
    # #mask abnormal_gray with edge map
    # abnormal_gray_ = org_gray * edge_map
    # #plot the histogram of abnormal gray_
    # plt.hist(abnormal_gray_.ravel(), 256, [0, 256])
    # plt.show()
    #
    # # Use weight map in Otsu's thresholding
    # abnormal_gray_ = abnormal_gray_.astype(np.uint8)
    # _, thresh = cv2.threshold(abnormal_gray_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # Apply threshold using _as threshold value
    # _, thresh = cv2.threshold(org_gray, _, 255, cv2.THRESH_BINARY)
    #show the result

    _,thresh=cv2.threshold(org_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('result', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(abnormal_img, markers)
    red = [255, 0, 0]

    abnormal_img[markers == -1] = red

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(markers, cmap="tab20b")
    # Draw unmatched keypoints

    for kp in unmatched_kp2:
        ax.plot(kp.pt[0], kp.pt[1], 'ro', markersize=6)
    ax.axis('off')
    plt.show()



    # Find the object with the most unmatched keypoints
    max_unmatched_keypoints = 0
    abnormal_object_index = 0
    # Convert unmatched_kp2 to an array of (x,y) coordinates
    unmatched_coords = np.float32([kp.pt for kp in unmatched_kp2]).reshape(-1, 1, 2)
    for label in range(1, np.max(markers)):
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == label] = 255

        unmatched_kp2_coords = [kp.pt for kp in unmatched_kp2]
        object_mask_pixels = np.where(mask == 255)

        num_unmatched_in_object = 0
        for kp_coord in unmatched_kp2_coords:
            x, y = kp_coord
            if mask[int(y), int(x)] == 255:
                num_unmatched_in_object += 1
        if num_unmatched_in_object > max_unmatched_keypoints:
            max_unmatched_keypoints = num_unmatched_in_object
            abnormal_object_index = label
    print("index:",abnormal_object_index,"max_unmatched_keypoints:",max_unmatched_keypoints)

    # Create a mask for the abnormal object
    abnormal_object_mask = (markers == abnormal_object_index).astype(np.uint8) * 255

    # Find the contours of the abnormal object
    contours, _ = cv2.findContours(abnormal_object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the abnormal object
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(abnormal_img, (x, y), (x + w, y + h), (0, 255, 0), 10)
    # show the result

    cv2.imshow('abnormal', abnormal_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return abnormal_img

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Read the normal and abnormal images
    normal_img = cv2.imread('Project2/0005_normal.jpg')
    abnormal_img = cv2.imread('Project2/0005_3.jpg')

    # Preprocessing (if necessary)



    # Apply any preprocessing steps here, such as resizing, denoising, or histogram equalization, if required.
    # create result folder
    if not os.path.exists('result'):
        os.mkdir('result')
    # Detect and square the abnormal object in the abnormal image
    result_img = abnoral_detection(normal_img, abnormal_img)

    # Save the result
    cv2.imwrite('result/00005_3_result.jpg', result_img)
