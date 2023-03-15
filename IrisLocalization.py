# import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def IrisLocalization(img_gray):

# parameters:
# img_gray: a gray scale eye image

# return:
# img_iris: the segmented eye image
# radii: an array storing the radius of each boundary circle (pupil and iris, respectively)
# centers: an array storing the centers of each boundary circle (pupil and iris, respectively)

# function: thresholding to segment the pupil, determining parameters, detecting circles on the image, and segmenting the iris to and output the image of the segmented iris along with the radius and center of each boundary circle (both stored in arrays)

    # convert image to binary for ease of projections
    # threshold set to one fourth of pixel intensity range to be selective
    img_binary = cv2.threshold(img_gray, 64, 255, cv2.THRESH_BINARY)[1]
    
    # project image vertically and horizontally, counting frequencies in arrays
    # initialize black pixel frequency counts per column or row (respectively) to 0
    height, width = img_binary.shape
    freq_vert = [0 for index in range(0, width)]
    freq_hor = [0 for index in range(0, height)]
    
    # traverse image pixel by pixel across rows and down columns
    # increment black pixel count when black pixel encountered
    for row in range(0, height):
        for col in range(0, width):
            if img_binary[row,col] == 0:
                freq_vert[col] += 1
                freq_hor[row] += 1
    
    # use projections to extimate the center of the pupil
    x_p = freq_vert.index(max(freq_vert))
    y_p = freq_hor.index(max(freq_hor))
    
    # isolate a 120 x 120 region around pupil in binary image
    up, down, left, right = [-60, 60, -60, 60]
    img_pupilRegion = img_binary[y_p+up:y_p+down, x_p+left:x_p+right]
    
    # expand region by 10 pixels in appropriate directions until entire pupil is contained within region
    cropped = True
    while cropped == True:
        cropped = False
        h_cropped, w_cropped = img_pupilRegion.shape
        if img_pupilRegion[0, w_cropped//2] == 0:
            up += -10
            cropped = True
        if img_pupilRegion[h_cropped-1, w_cropped//2] == 0:
            down += 10
            cropped = True
        if img_pupilRegion[h_cropped//2, 0] == 0:
            left += -10
            cropped = True
        if img_pupilRegion[h_cropped//2, w_cropped-1] == 0:
            right += 10
            cropped = True

        img_pupilRegion = img_binary[y_p+up:y_p+down, x_p+left:x_p+right]
        
    # note: img_pupilRegion was originally analized as grayscale image
    # then histogram was viewed using plt.hist(img_pupilRegion.ravel(), 256, [0,256])
    # based on the histogram, the pupil segmentation threshold was set to 75
    # because this code is not needed after initial parameter setting, it is not included here
    
    # use median blur to lessen noise from eyelashes
    # ksize of 7 results in sufficient blur to remove most significant noise
    img_medianBlurred = cv2.medianBlur(img_pupilRegion, ksize = 7)
    
    # invert binary image to find pupil centroid
    img_inverted = np.invert(img_medianBlurred)
    
    # find centroid of inverted segmented pupil
    # compute moments, then the coordinates of the center of the pupil
    # if division by zero is attempted, keep the pupil center approximation from the histograms (represented in the cropped image as the number of pixels from the left and upper boundaries of the image)
    moments = cv2.moments(img_inverted)
    if moments["m00"] == 0:
        x_c = -right
        y_c = -up
    else:
        x_c = int(moments["m10"] / moments["m00"])
        y_c = int(moments["m01"] / moments["m00"])
    
    # uncomment below to visually check the accuracy of pupil center detection:
    # img_pupilCenter = cv2.rectangle(img_pupilRegion.copy(), (x_c-3,y_c-3), (x_c+3,y_c+3), 255, 2)
    # cv2.imshow(imgFilename, img_pupilCenter)
    
    # approximate the radius of the pupil for use cleaning noise after edge detection and narrowing search for pupil and iris boundary circle size
    # starting at the approximated pupil center, compute distance to edge of pupil to the left and right (where there is unlikely to be eyelid interruption)
    rad = [None] * 2
    step = [-1,1]
    for s in step:
        hor_pos = x_c
        while img_pupilRegion[y_c,hor_pos] == 0:
            hor_pos += s
        rad[step.index(s)] = abs(hor_pos - y_c)

    # average the values to obtain the approximate pupil radius
    r_avg = (rad[0] + rad[1])//2
    
    # return to uncropped image, shifting pupil center coordinates appropriately
    x_c += (x_p + left)
    y_c += (y_p + up)
    
    # prepare for edge detection
    # use median blur to reduce noise
    # parameter (5,5) allows for moderate blur that disregards finer details, but preserves necessary edges like pupil and iris boundaries
    img_GaussianBlur = cv2.GaussianBlur(img_gray.copy(), (5, 5), 0)
    
    # detect edges using Canny operator
    # parameters were set by observation/trial and error
    # lower parameters that are closer together result in more detailed edges, which are necessary for the outer boundary of iris to be detected
    # with other parameter choices, either there was 1) too many edges due to iris and eyelash details, and/or 2) the necessary edges did not appear, namely the weak iris outer boundary
    img_edges = cv2.Canny(img_GaussianBlur, 20, 30)
    
    # manually remove noise based on approximate pupil radius
    # this is based on the assumption that the radius approximated earlier is nearly accurate, and that the iris outer boundary can be expected to have a radius between 2 and 3.2 times the radius of the pupil (by observation)
    for col in range(0, width):
        for row in range(0, height):
            if ((col-x_c)**2 + (row-y_c)**2 >= (3.2*r_avg)**2):
                img_edges[row,col] = 0
            if ((col-x_c)**2 + (row-y_c)**2 >= (1.2*r_avg)**2) and ((col-x_c)**2 + (row-y_c)**2 <= (2*r_avg)**2):
                img_edges[row,col] = 0
    
    # use Hough transform twice: once for pupil boundary detection, once for outer iris boundary detection
    # the transform is performed twice to avoid competing concentric circles
    # store the resulting radii in an array [pupil boundary, iris boundary]
    # display resulting boundary circles on a copy of the original grayscale image
    # other parameters are chosen by observed success/trial and error
    img_irisLocation = img_gray.copy()
    centers = [None] * 2
    radii = [None] * 2
    params = [[r_avg-15, r_avg+15], [2*r_avg, np.uint8(np.around(4*r_avg))]]
    for p in params:

        circles = cv2.HoughCircles(img_edges, cv2.HOUGH_GRADIENT, 1, 10, param1 = 50,
                                param2 = 10, minRadius = p[0], maxRadius = p[1])
        
        # parameters in Hough transform function have narrowed circle options significantly; only circles with radii in the above expected intervals are considered
        # average the center coordinates and radii of the circles that fit the parameters to approximate the pupil boundary and the outer boundary of the iris during the first and second calls, respectively
        if circles is not None:
            x_avg = 0
            y_avg = 0
            rad_avg = 0
            count = 0
            circles = np.uint8(np.around(circles))
            for c in circles[0,:]:
                # consider only those circles with centers near the approximated center of the pupil
                if (x_c-10 <= c[0] <= x_c+10) and (y_c-10 <= c[1] <= y_c+10):
                    x_avg += c[0]
                    y_avg += c[1]
                    rad_avg += c[2]
                    count += 1
                    
            if count > 0:
                # at least one value circle was discovered; use circle information to compute center coordinates and radii
                x_avg = np.uint8(np.around(x_avg/count))
                y_avg = np.uint8(np.round(y_avg/count))
                rad_avg = np.uint8(np.around(rad_avg/count))
                # for visualization purposes, uncomment to display discovered pupil and iris outer boundary circles
                # cv2.circle(img_irisLocation, (x_avg, y_avg), rad_avg, 255, 2)
                centers[params.index(p)] = [x_avg,y_avg]
                radii[params.index(p)] = rad_avg
                
            else:
                # no circles were detected; to allow the program to continue to run, set center coordinates and each radius according to the radius approximated above
                centers[params.index(p)] = [x_c,y_c]
                if params.index(p) == 0:
                    radii[0] = r_avg
                else:
                    radii[1] = np.uint8(np.around(2.5*r_avg))
                
    # segment the iris based on detected circle radii
    # this is based on the approximate pupil center, for both boundary circles are known to be close to this pixel location
    img_iris = img_gray.copy()

    for col in range(0, width):
        for row in range(0, height):
            if ((col-centers[0][0])**2 + (row-centers[0][1])**2 <= radii[0]**2):
                img_iris[row,col] = 0
            if ((col-centers[1][0])**2 + (row-centers[1][1])**2 >= radii[1]**2):
                img_iris[row,col] = 0

    # return the image with segmented iris, boundary circle radii, and boundary circle centers
    return img_iris, centers, radii


