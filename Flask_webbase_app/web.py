from flask import Flask, render_template, request, send_from_directory, redirect
import glob, os
from werkzeug.utils import secure_filename
import numpy as np
import argparse

import cv2
import os
import re
import threading

from flask import flash

import sys
import random as rd

import tensorflow as tf
import pytesseract
from lib_detection import load_model, detect_lp, im2single,normal,reconstruct
import time


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
# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu
# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

#load pretrain model




#Find LP_h + w

def find_ratio_lp(Ivehicle):
    # Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
    Dmax = 608
    Dmin = 288
    print("h3")
    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    print("h3")
    
    #_ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)
    #detect_lp(model, I, max_dim, lp_threshold):
    I = im2single(Ivehicle)
    
    max_dim = bound_dim
    lp_threshold=0.5
    # Tính factor resize ảnh
    min_dim_img = min(I.shape[:2])
    factor = float(max_dim) / min_dim_img

    # Tính W và H mới sau khi resize
    w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()

    # Tiến hành resize ảnh
    Iresized = cv2.resize(I, (w, h))

    T = Iresized.copy()

    # Chuyển thành Tensor
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    wpod_net_path = "wpod-net_update1.json"
    wpod_net = load_model(wpod_net_path)
    # Tiến hành detect biển số bằng Wpod-net pretrain
    model = wpod_net
    Yr = model.predict(T)

    # Remove các chiều =1 của Yr
    Yr = np.squeeze(Yr)

    print(Yr.shape)

    # Tái tạo và trả về các biến gồm: Nhãn, Ảnh biến số, Loại biển số (1: dài: 2 vuông)
    _ , LpImg, lp_type = reconstruct(I, Iresized, Yr, lp_threshold)





    if (len(LpImg)):

        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        roi = LpImg[0]

        LpImg_shape = roi.shape
        Lp_h = LpImg_shape[0]
        Lp_w = LpImg_shape[1]
    
        Lp_ratio = float(Lp_w)/float(Lp_h)
        print(str(Lp_ratio))
        return roi, Lp_ratio, Lp_w, Lp_h
    else:
        return Ivehicle, -1, -1, -1

def predict(img_plate):
    model_svm = cv2.ml.SVM_load('svm2.xml')
    gray = cv2.cvtColor( img_plate, cv2.COLOR_BGR2GRAY)

    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY_INV)[1]

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
            
            if h/img_plate.shape[0]>=0.55: # Chon cac contour cao tu 60% bien so tro len
                #print(ratio)
                #print(h/img_plate.shape[0])
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

                plate_info +=result
    return plate_info
#webFunction
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['jpg'])


DEBUG = True

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/license", methods=['POST'])

def image_search():
    #Get data from form
   
    if request.method == 'POST':
        if 'file_original' not in request.files:
            flash('Images type wrong')
            return redirect(request.url)

        file1 = request.files['file_original']
        if file1.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file1 and allowed_file(file1.filename):
            filename1 = secure_filename(file1.filename)
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))

     
        #End: Get data from form
        path=[]
        files=[]
        #Start: process
        if filename1.find(".jpg") != -1:
            #Load img
            Ivehicle = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            #Save img for display to compare
            cv2.imwrite(os.path.sep.join(["static/images", "original_"+filename1]),Ivehicle)
            path.append(os.path.sep.join(["static/images", "original_"+filename1]))
            #find ratio
            result = find_ratio_lp(Ivehicle)
            if result[1] >= 2:
            #predict as oto
                str_plate = str(predict(result[0]))
            else:
                Lp_w = result[2]
                Lp_h = result[3]
                a = []
                temp = result[0]
                for i in range(0,2):
                    a.append(0)
                a[0]=temp[0:int(Lp_h/2), 0:Lp_w]
                a[1]=temp[1+int(Lp_h/2):Lp_h-1,0:Lp_w]
                str_plate = str(predict(a[0]))+" "+str(predict(a[1]))
            #print(str_plate)

            cv2.imwrite(os.path.sep.join(["static/images", "result"+filename1]),result[0])
            path.append(os.path.sep.join(["static/images", "result"+filename1]))
            
            files.append(glob.glob(path[0]))
            files.append(glob.glob(path[1]))
            return render_template('result.html', files=files, path=path, value=str_plate)
        #End: process


#print(result2d)
       


        
        
        #files.append(glob.glob(path[0]))
        #return render_template('image-search.html', files=files, path=path)
    


@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory('images/', path)

if __name__ == "__main__":
	app.run()
