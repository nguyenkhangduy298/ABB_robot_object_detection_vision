import argparse
import os

from Utils.Lib import *

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--LeftCamera', required=True, help='Left Camera: Port')
ap.add_argument('-r', '--RightCamera', required=True, help='Right Camera: Port')
ap.add_argument('-lf', '--LeftCameraFolder', required=True, help='Directory to the folder of Left Camera\'s Pictures')
ap.add_argument('-rf', '--RightCameraFolder', required=True, help='Directory to the folder of Right Camera\'s Pictures')

ap.add_argument('-c', '--counter', required=True, help="Initial counter")

ap.add_argument('-cw', '--ChessWidth', default=9, required=False, help="Width of the chess board, default: 9")
ap.add_argument('-ch', '--ChessHeight', default=6, required=False, help="Height of the chess board, default: 6")

args = ap.parse_args()

CAMERA_PORT1 = int(args.LeftCamera)
CAMERA_PORT2 = int(args.RightCamera)
CAMERA1_FOLDER = str(args.LeftCameraFolder)
CAMERA2_FOLDER = str(args.RightCameraFolder)
COUNT_BEGIN = int(args.counter)

# Check if folder exist
if not os.path.exists(CAMERA1_FOLDER):
    print("Not exist, create:" + CAMERA1_FOLDER)
    os.makedirs(CAMERA1_FOLDER)
if not os.path.exists(CAMERA2_FOLDER):
    print("Not exist, create:" + CAMERA2_FOLDER)
    os.makedirs(CAMERA2_FOLDER)

# Check if arguement for chess board's width and length exist, if not, use default value (6 and 9)
width = int(args.ChessWidth)
height = int(args.ChessHeight)
CHESS_SIZE = {
    "WIDTH": width,
    "HEIGHT": height
}

# Set VideoCapture to choosen ports
cap1 = cv2.VideoCapture(CAMERA_PORT1)
cap2 = cv2.VideoCapture(CAMERA_PORT2)


# Small function to create text on frame
def drawText(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while (True):
    _, img1 = cap1.read()
    _, img2 = cap2.read()

    original1 = img1.copy()
    original2 = img2.copy()

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret1, corners_img1 = cv2.findChessboardCorners(gray_img1, (CHESS_SIZE["WIDTH"], CHESS_SIZE["HEIGHT"]), None)
    ret2, corners_img2 = cv2.findChessboardCorners(gray_img2, (CHESS_SIZE["WIDTH"], CHESS_SIZE["HEIGHT"]), None)
    # If found, add object points, image points (after refining them)
    if ret1 == True & ret2 == True:
        corners2L = cv2.cornerSubPix(gray_img2, corners_img2, (11, 11), (-1, -1), criteria)
        corners2R = cv2.cornerSubPix(gray_img1, corners_img1, (11, 11), (-1, -1), criteria)  # Refining the Position

        cv2.drawChessboardCorners(img1, (CHESS_SIZE["WIDTH"], CHESS_SIZE["HEIGHT"]), corners2R, ret1)
        cv2.drawChessboardCorners(img2, (CHESS_SIZE["WIDTH"], CHESS_SIZE["HEIGHT"]), corners2L, ret2)

    numpy_horizontal = np.hstack((img1, img2))
    cv2.imshow("Camera 1 and 2", numpy_horizontal)

    # Capture The Images:

    # Capture Trigger by letter key "e"
    if cv2.waitKey(1) & 0xFF == ord('e'):
        print("Images Taken: " + str(COUNT_BEGIN))
        cv2.imwrite(CAMERA1_FOLDER + str(COUNT_BEGIN) + ".png", original1)
        cv2.imwrite(CAMERA2_FOLDER + str(COUNT_BEGIN) + ".png", original2)
        COUNT_BEGIN += 1
    elif cv2.waitKey(2) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap1.release()
cap2.release()
