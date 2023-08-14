Project Description:
Detecting Abnormal Objects in Images Using SIFT Algorithm

Welcome to our image analysis project that addresses the complexities posed by diverse lighting conditions and camera angles in object detection. Our approach utilizes the Scale-Invariant Feature Transform (SIFT) algorithm to reliably identify abnormal objects within images, overcoming challenges related to lighting variations and perspective shifts.

Key Features:

Robust Abnormal Object Detection: Our method excels in locating abnormal objects irrespective of changing lighting conditions and camera angles. It leverages the power of the SIFT algorithm to ensure accurate identification.
Methodology:
To achieve accurate and consistent results, we follow a multi-step methodology:

Preprocessing: We employ Gaussian blur and histogram equalization to enhance image quality, reducing noise and minor inconsistencies.
SIFT Feature Extraction: The SIFT algorithm is utilized to extract distinctive keypoints and descriptors from both normal and abnormal images. Specialized settings are employed to capture key points of interest.
Feature Matching: Utilizing the BFMatcher, feature descriptors from normal and abnormal images are matched, creating correspondences between salient features.
Segmentation: Abnormal images undergo segmentation using the Felzenszwalb algorithm, isolating distinct regions of interest based on intensity levels.
Identification of Abnormal Regions: Unmatched key points from matched pairs are isolated. These key points correspond to regions with abnormal characteristics, deviating from the standard appearance.
Abnormal Object Localization: The segment with the highest number of unmatched key points is identified as the abnormal object of interest. A specific mask is created for the identified abnormal object.

Licensing:
Our code is available for non-commercial use, with or without modifications, as long as the source is properly indicated.
