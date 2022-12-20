from scipy import signal
import cv2
import numpy as np
import math
from tkinter import *


def Point():
    # путь к указанному входному изображению и
    # изображение загружается с помощью команды imread
    image = cv2.imread('labs/book3.jpg')

    # конвертировать входное изображение в Цветовое пространство в оттенках серого
    operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # изменить тип данных
    # установка 32-битной плавающей запятой
    operatedImage = np.float32(operatedImage)

    # применить метод cv2.cornerHarris
    # для определения углов с соответствующими
    # значения в качестве входных параметров
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    # Результаты отмечены через расширенные углы
    dest = cv2.dilate(dest, None)

    # Возвращаясь к исходному изображению,
    # с оптимальным пороговым значением
    image[dest > 0.0001 * dest.max()] = [255, 0, 0]

    # окно с выводимым изображением с углами
    cv2.imshow('Image with Borders', image)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def Rotate():
    
    cap = cv2.VideoCapture('girl.mp4')
    
    # params for corner detection
    feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7,  blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03))
    
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    while(1):
        
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray,p0, None, **lk_params)
    
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new,  good_old)):
            a, b = new.ravel()
            a=math.floor(a)
            b=math.floor(b)
            c, d = old.ravel()
            c=math.floor(c)
            d=math.floor(d)
            mask = cv2.line(mask, (a, b), (c, d),  color[i].tolist(), 2)
            
            frame = cv2.circle(frame, (a, b), 5,color[i].tolist(), -1)
            
        img = cv2.add(frame, mask)
    
        cv2.imshow('frame', img)
        
        k = cv2.waitKey(25)
        if k == 27:
            break
    
        # Updating Previous frame and points 
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    cv2.destroyAllWindows()
    cap.release()


def Find():
    query_img = cv2.imread('poker1.png')

    original_img = cv2.imread('poker.png') 


    query_img_bw = cv2.cvtColor(query_img, cv2.IMREAD_GRAYSCALE)
    original_img_bw = cv2.cvtColor(original_img, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()

    queryKP, queryDes = orb.detectAndCompute(query_img_bw,None)
    trainKP, trainDes = orb.detectAndCompute(original_img_bw,None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(queryDes,trainDes)

    matches = sorted(matches, key = lambda x:x.distance)

    final_img = cv2.drawMatches(query_img, queryKP, original_img, trainKP, matches[:20],None)
    
    final_img = cv2.resize(final_img, (1000,650))

    cv2.imshow("Matches", final_img)
    cv2.waitKey()


def Menu():
    window = Tk()

    
    window.title("Menu")

    w = window.winfo_screenwidth()
    h = window.winfo_screenheight()
    w = w//2 # середина экрана
    h = h//2 
    w = w - 200 # смещение от середины
    h = h - 200
    window.geometry('300x300+{}+{}'.format(w, h))
    window.configure(bg='#bb85f3')

    btn = Button(window, text="Нахождение точек", padx=5, pady=5, command = Point, bg='#eec6ea')  
    btn.pack(anchor="center", padx=50, pady=20)


    btn1 = Button(window, text="Сравнение точек", padx=5, pady=5, command =Find, bg='#eec6ea')  
    btn1.pack(anchor="center", padx=50, pady=20)

    btn2 = Button(window, text="Оптический поток", padx=5, pady=5, command =Rotate, bg='#eec6ea')  
    btn2.pack(anchor="center", padx=50, pady=20)

    btn3 = Button(window, text="Выход", padx=5, pady=5, command =exit, bg='#eec6ea')  
    btn3.pack(anchor="center", padx=50, pady=20)
    


    window.mainloop()

Menu()