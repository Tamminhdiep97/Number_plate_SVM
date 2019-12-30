import pytesseract
import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single
import time
import os
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

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
#img_path = "test/41080.jpg"

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)
model_svm = cv2.ml.SVM_load('svm2.xml')
# Đọc file ảnh đầu vào

directory = "test/"
for filename in os.listdir(directory):
    img_path = os.path.join(directory, filename)
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



    if (len(LpImg)):

        # Chuyen doi anh bien so
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        roi = LpImg[0]


        LpImg_shape = roi.shape
        Lp_h = LpImg_shape[0]
        Lp_w = LpImg_shape[1]
        
        Lp_ratio = float(Lp_w)/float(Lp_h)
        print(Lp_w,Lp_h)
        #Neu ti le w/h cua bien so >=2 thi la oto, nguoc lai la xe may 
        if Lp_ratio >= 2:
        # Chuyen anh bien so ve gray
            gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)


        # Ap dung threshold de phan tach so va nen
            binary = cv2.threshold(gray, 127, 255,
                            cv2.THRESH_BINARY_INV)[1]

    #    cv2.imshow("Anh bien so sau threshold", binary)
    #    cv2.waitKey()

        # Segment kí tự
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            im2,cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


            plate_info = ""

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w

                #print(ratio)
                if 0.9<=ratio<=4.5: # Chon cac contour dam bao ve ratio w/h
                
                    if h/roi.shape[0]>=0.55: # Chon cac contour cao tu 60% bien so tro len

                    # Ve khung chu nhat quanh so
                        #cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Tach so va predict
                        curr_num = thre_mor[y:y+h,x:x+w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                        #cv2.imshow("i",curr_num)
                        #cv2.waitKey(0)
                        curr_num = np.array(curr_num,dtype=np.float32)
                        curr_num = curr_num.reshape(-1, digit_w * digit_h)
                    
                    # Dua vao model SVM
                        result = model_svm.predict(curr_num)[1]
                        result = int(result[0, 0])
        
                        if result<=9: # Neu la so thi hien thi luon
                            result = str(result)
                        else: #Neu la chu thi chuyen bang ASCII
                            result = chr(result)
                    
                    #print(result)
                        plate_info +=result

            #cv2.imshow("Cac contour tim duoc", roi)
            #cv2.waitKey(0)

        # Viet bien so len anh
            #cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

        # Hien thi anh
            print("Bien so=", plate_info)
            cv2.imshow("Hinh anh output",roi)
            cv2.waitKey()

        else:
            temp = LpImg[0]
            a = []
            plate_info = ""
            for i in range(0,2):
                a.append(0)
            
            a[0]=temp[0:int(Lp_h/2), 0:Lp_w]
            a[1]=temp[1+int(Lp_h/2):Lp_h-1,0:Lp_w]
            for i in range(0,2):
            
                roi = a[i]

                #cv2.imshow(str(i),roi)
                #cv2.waitKey(0)
        # Chuyen anh bien so ve gray
                gray = cv2.cvtColor( roi, cv2.COLOR_BGR2GRAY)


        # Ap dung threshold de phan tach so va nen
                binary = cv2.threshold(gray, 127, 255,
                            cv2.THRESH_BINARY_INV)[1]

    #    cv2.imshow("Anh bien so sau threshold", binary)
    #    cv2.waitKey()

        # Segment kí tự
                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
                im2,cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


                

                for c in sort_contours(cont):
                    (x, y, w, h) = cv2.boundingRect(c)
                    ratio = h/w
            #print(ratio)
                    if 0.9<=ratio<=4.5: # Chon cac contour dam bao ve ratio w/h
                
                        if h/roi.shape[0]>=0.55: # Chon cac contour cao tu 60% bien so tro len

                    # Ve khung chu nhat quanh so
                            #cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Tach so va predict
                            curr_num = thre_mor[y:y+h,x:x+w]
                            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                            _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                            #cv2.imshow("i",curr_num)
                            #cv2.waitKey(0)
                            curr_num = np.array(curr_num,dtype=np.float32)
                            curr_num = curr_num.reshape(-1, digit_w * digit_h)
                    
                    # Dua vao model SVM
                            result = model_svm.predict(curr_num)[1]
                            result = int(result[0, 0])
        
                            if result<=9: # Neu la so thi hien thi luon
                                result = str(result)
                            else: #Neu la chu thi chuyen bang ASCII
                                result = chr(result)
                    
                    #print(result)
                            plate_info +=result
                plate_info+=" "
            print("Bien so = ",plate_info)
            cv2.imshow("Lp_crop",LpImg[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()           
                #cv2.imshow("Cac contour tim duoc", roi)
                #cv2.waitKey(0)

        # Viet bien so len anh
            """cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

        
            """













    #cv2.destroyAllWindows()