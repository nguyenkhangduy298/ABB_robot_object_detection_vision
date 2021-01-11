import cv2

from Utils import StereoTools
from Utils.Lib import findCorresPixelCoordinateFromObject, getDispAtPoint


def DepthImageProcessing(time, leftCap, rightCap, Left_Stereo_Map, Right_Stereo_Map, wls_filter,
                         min_disp, num_disp, stereoR, stereo, detectedObj, mtx, dis, P, R, max_depth=100, min_depth=40,
                         threshold_check=True):
    depth_values = 0
    n = 0
    while (True):
        ret1, nextLeftFrame = leftCap.read()
        ret2, nextRightFrame = rightCap.read()
        Left_nice = cv2.remap(nextLeftFrame, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT, 0)
        Right_nice = cv2.remap(nextRightFrame, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,
                               cv2.BORDER_CONSTANT, 0)
        (filteredImg, disparityMap) = StereoTools.returnDisparityMap(Left_nice=Left_nice, Right_nice=Right_nice,
                                                                     wls_filter=wls_filter, min_disp=min_disp,
                                                                     num_disp=num_disp, stereoR=stereoR, stereo=stereo)
        (x_rectified, y_rectified) = findCorresPixelCoordinateFromObject(object=detectedObj,
                                                                         mtxL=mtx, disL=dis, P=P, R=R)
        depth = getDispAtPoint(x=x_rectified + detectedObj.width / 2,
                               y=y_rectified + detectedObj.height / 2,
                               disp=disparityMap)
        if (threshold_check == True and depth <= max_depth and depth >= min_depth):
            depth_values += depth
            n += 1
        if (threshold_check == False):
            depth_values += depth
            n += 1
        if (n >= time):
            break
    return depth_values / time


def XyzCalculate():
    pass



if __name__ == '__main__':
    array_depth = [44, 45, 45, 44, 44, 44, 45]
    # pollingDepth(array_depth=array_depth)
