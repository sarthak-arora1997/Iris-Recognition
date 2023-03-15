# Machine Learning for Image Analysis (Group Project)
Using the provided set of eye images from the CASIA Iris Image Database (version 1.0) we built an iris recognition model using Specific methods, parameter recommendations, and other details of the process were gleaned from the provided paper [**Personal Identification Based on Iris Texture Analysis** by Li Ma, et al. (2003)](https://www.mukpublications.com/resources/ijcvb2-1-12.pdf)

## Authors
Sarthak Arora, Erin Josephine Donnelly, Yo Jeremijenko-Conley

Overview
--------
The code in IrisRecognition.py (with subfunctions IrisLocalization, IrisNormalization, ImageEnhancement, FeatureExtraction, IrisMatching, and PerformanceEvaluation) implements iris localization and matching as detailed in the assignment instructions provided in GroupProject_Fall2021.pdf. Using the provided set of eye images from the CASIA Iris Image Database (version 1.0), the program successfully localizes the iris bounded by the pupil and the outer boundary of the iris, normalizes the iris using polar coordinates, enhances the image using histogram equalization, extracts iris features, and matches eye images from a training set to those of the testing set with an average above *75% accuracy*.


Program Logic
-------------
The logic of the program is explained directly in the main .py script and in the subfunctions .py scripts in the form of comments above and throughout blocks of code.


Variables and Parameters
------------------------
Variables and parameters are explain directly in the .py scripts in the form of comments above and throughout the script. Variables names are descriptive and are often named according to the most recent modification applied to the image. Matrices corresponding to images are prefaced by "img_", while matrices and other variables that do not directly represent images are not.


Limitations and Improvements
----------------------------
Some current limitations include:
* The program does not implement quality assessment and image selection/rejection, as it is not a part of the assignment expectations; we are trusting that the database contains only quality eye images from which clear features may be extracted.
* Many eye images have very occluded irises; the upper and/or lower eye lids obscure the iris and cover its pattern, sometimes with eyelashes providing significant noise that interfere with accurate feature extraction.
* The program does not verify the validity of eye images, including ensuring the image is a real eye without contacts overlaying an artificial iris pattern.
* The program was designed to produce successful matching results for the eye images in this database. Components of the program (example: "unrolling" the iris image to a 64x512 image during normalization) are hardcoded and may not be effective if the program were to be used with other images.
* Some parameters were chosen by observation as opposed to by some rigorous or machine learning measure.
* Some decisions had to be made in IrisLocalization to account circumstances that would result in errors. For example, if an iris circle passed the boundaries of the image, the code assigned pixel values of 0 during normalization, which is acceptable because there are no features to detect outside the bounds of the image anyway.
* Also in IrisLocalization, following edge detection, the outer boundary of the iris is not a strong circular presence. As a result, many circles are found that fit the parameters for the pupil and iris during the Hough transforms. The solution in this program---computing the averages of the circles' x-coordinates, y-coordinates, and radii, respectively---is sufficient, but it would be nice to have a more definitive way to identify the iris outer boundary and pupil with a single circle.
* A few methods and suggestions from the Li Ma paper that did not improve the accuracy of our matching program. Most significantly, in ImageEnhancement, the background illumination removal and histogram equalization traversing the image by blocks of 32x32 pixels were implemented successfully, but distorted the image in a way that generated intensity discrepancies, which could be misinterpreted as false features. Thus some features may be less distinct due to the histogram equalization being performed over the entire normalized image.
* Additionally, it did not improve the accuracy of the matching program to segment noise such as glare and eyelashes/eyelids and set their pixels to zero before or after enhancement; similarly to the above note, this generated edges that may be incorrectly interpreted as features.
    

Some modifications that could be made to improve performance include:
* Acquire eye images that are unobstructed by eyelids. This is unrealistic in many practical applications, but the reduced noise would certainly improve feature extraction and pattern matching.
* In IrisLocalization, find or create an edge detection method that is more reliable in its detection of the outer boundary of the iris.
* In IrisNormalization, handle out-of-bounds iris circles in a more refined manner than assigning pixel values to 0 in normalized image.
* In ImageEnhancement, implement a block-by-block histogram equalization to remove background illumination and enhance the image in a way that does not result in falsely identified features.
* Also in ImageEnhancement, remove glare, eyelashes, eyelids before detecting features. Eyelid and (perhaps partial eyelash) removal may be accomplished by fitting parabolic boundaries to segment the iris fron the eyelids.
* In IrisMatching, shift the enhanced normalized images' features to account for slight rotations in the iris image due to natural tilting of the head.
* As is the case in many experiments, more testing and training examples gives more insight into program training and may result in improved results.
* To make this program suitable for use beyond this dataset, certain parameters, dimensions, etc. will need to be learned instead of hardcoded.
