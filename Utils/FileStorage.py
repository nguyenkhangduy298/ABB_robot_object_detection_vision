import cv2
import xlsxwriter


def SaveCalibration(destination, MLS, dLS, MRS, dRS, R, T, shapeWidth, shapeHeight, mtxL, mtxR, disL, disR, OpMatL,
                    OpMatR):
    try:
        cv_file = cv2.FileStorage(destination, cv2.FILE_STORAGE_WRITE)

        cv_file.write("MLS", MLS)
        cv_file.write("MRS", MRS)
        cv_file.write("dLS", dLS)
        cv_file.write("dRS", dRS)

        cv_file.write("RotationalMatrix", R)
        cv_file.write("FundamentalMatrix", T)
        cv_file.write("width", int(shapeWidth))
        cv_file.write("height", int(shapeHeight))
        cv_file.write("mtxL", mtxL)
        cv_file.write("mtxR", mtxR)
        cv_file.write("disL", disL)
        cv_file.write("disR", disR)
        cv_file.write("OpMatL", OpMatL)
        cv_file.write("OpMatR", OpMatR)

        # cv_file.write("rvecR", rvecsR)
        # cv_file.write("tvecR", tvecsR)
        # cv_file.write("rvecL", rvecsL)
        # cv_file.write("tvecL", tvecsL)

        cv_file.release()
        return "File Save Finished"
    except Exception:
        return "Problem Happened"

def SaveSingleCalibration(destination,  mtxL, disL):
    try:
        cv_file = cv2.FileStorage(destination, cv2.FILE_STORAGE_WRITE)
        cv_file.write("MTX", mtxL)
        cv_file.write("DIS", disL)
        # cv_file.write("RVEC", rvecsL)
        # cv_file.write("TVEC", tvecsL)
        cv_file.release()
        return "File Save Finished"
    except Exception:
        return "Problem Happened"

def ReadSingleCalib(read_path):
    cv_file = cv2.FileStorage(read_path, cv2.FILE_STORAGE_READ)

    MTX = cv_file.getNode("MTX").mat()
    DIS = cv_file.getNode("DIS").mat()
    # RVEC = cv_file.getNode("RVEC").mat()
    # TVEC = cv_file.getNode("TVEC").mat()
    return (MTX, DIS)

def ReadCalibration(read_path):
    cv_file = cv2.FileStorage(read_path, cv2.FILE_STORAGE_READ)

    MLS = cv_file.getNode("MLS").mat()
    MRS = cv_file.getNode("MRS").mat()
    dLS = cv_file.getNode("dLS").mat()
    dRS = cv_file.getNode("dRS").mat()
    R = cv_file.getNode("RotationalMatrix").mat()
    T = cv_file.getNode("FundamentalMatrix").mat()
    width = cv_file.getNode("width").real()
    height = cv_file.getNode("height").real()
    mtxR = cv_file.getNode("mtxR").mat()
    mtxL = cv_file.getNode("mtxL").mat()
    disL = cv_file.getNode("disL").mat()
    disR = cv_file.getNode("disR").mat()
    OpMatL = cv_file.getNode("OpMatL").mat()
    OpMatR = cv_file.getNode("OpMatR").mat()

    # tvecR = cv_file.getNode("tvecR").mat()
    # rvecR = cv_file.getNode("rvecR").mat()
    # rvecL = cv_file.getNode("rvecL").mat()
    # tvecL = cv_file.getNode("tvecL").mat()


    cv_file.release()
    CalibrationParameters = {
        "MLS": MLS,
        "MRS": MRS,
        "dLS": dLS,
        "dRS": dRS,
        "R": R,
        "T": T,
        "width": width,
        "height": height,
        "disL": disL,
        "disR": disR,
        "mtxR": mtxR,
        "mtxL": mtxL,
        "OpMatL": OpMatL,
        "OpMatR": OpMatR,


    }

    # Test the different between file saved and file written:
    return CalibrationParameters


def writeXLSX(file_path, data=()):
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet()
    # Start from the first cell. Rows and columns are zero indexed.
    row = 0
    col = 0

    # Iterate over the data and write it out row by row.
    for item, cost in (data):
        worksheet.write(row, col, item)
        worksheet.write(row, col + 1, cost)
        row += 1
    workbook.close()


if __name__ == '__main__':
    ReadCalibration("duyComputer.yaml")
