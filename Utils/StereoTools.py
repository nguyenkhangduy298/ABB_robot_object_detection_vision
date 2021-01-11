import cv2
import numpy


# *******************************************
# ***** Parameters for the StereoVision *****
# *******************************************

# minDisparity – Minimum possible disparity value.
# numDisparities – Maximum disparity minus minimum disparity. This parameter must be divisible by 16.
# SADWindowSize – Matched block size. It must be an odd number >=1 .
# disp12MaxDiff – Maximum allowed difference (in integer pixel units) in the left-right disparity check.
# preFilterCap – Truncation value for the prefiltered image pixels.
# uniquenessRatio – Margin in percentage by which the best (minimum) computed cost function value should “win”
# the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
# speckleWindowSize – Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# speckleRange – Maximum disparity variation within each connected component
# Create StereoSGBM and prepare all parameters

def returnStereoSGBM(window_size=3, min_disp=0, max_disp=130, lmbda=80000, sigma=1.8,
                     visual_multiplier=1.0, uniquesnessRatio=10, speckleWindowSize=100, speckleRange=32,
                     disp12MaxDiff=5):
    num_disp = max_disp - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size,
                                   uniquenessRatio=uniquesnessRatio,
                                   speckleWindowSize=speckleWindowSize,
                                   speckleRange=speckleRange,
                                   disp12MaxDiff=disp12MaxDiff,
                                   P1=8 * 3 * window_size ** 2, P2=32 * 3 * window_size ** 2)
    # Used for the filtered image
    stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time
    stereoL = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    Dict = {
        "StereoSGBM": stereo,
        "StereoR": stereoR,
        "StereoL": stereoL,
        "WLS_Filter": wls_filter,
        "min_disp": min_disp,
        "num_disp": num_disp

    }
    return Dict


def returnDisparityMap(Right_nice, Left_nice, stereo, wls_filter, stereoR, min_disp, num_disp):
    grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
    #
    disp = stereo.compute(grayL, grayR)
    dispL = disp
    dispR = stereoR.compute(grayR, grayL)
    #
    # USING FILTERED IMAGE
    filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg,
                                beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

    filteredImg = numpy.uint8(filteredImg)
    disp = ((disp.astype(numpy.float32) / 16) - min_disp) / num_disp

    return (filteredImg, disp)
