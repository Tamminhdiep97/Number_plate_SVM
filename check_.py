import os
import pytesseract
import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single


# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):

    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

directory = "Phu/train/"
for filename in os.listdir(directory):
    img_path = os.path.join(directory, filename)
    # Load model LP detection
    wpod_net_path = "wpod-net_update1.json"
    wpod_net = load_model(wpod_net_path)

# Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(img_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
    Dmax = 608
    Dmin = 288

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)


# Cau hinh tham so cho model SVM
    digit_w = 30 # Kich thuoc ki tu
    digit_h = 60 # Kich thuoc ki tu

    model_svm = cv2.ml.SVM_load('svm.xml')

    if (len(LpImg)):

    # Chuyen doi anh bien so
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        roi = LpImg[0]

    # Chuyen anh bien so ve gray
        gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)


    # Ap dung threshold de phan tach so va nen
        binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY_INV)[1]

        cv2.imshow(filename, binary)
        print(filename)
        if cv2.waitKey() & 0xFF==ord('q') :
            break
        cv2.destroyAllWindows()
