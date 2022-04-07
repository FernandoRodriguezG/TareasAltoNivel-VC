import cv2
import numpy as np
import os

def descriptores(images):
    desList=[]
    for i in images:
        kp,des = orb.detectAndCompute(i,None)
        desList.append(des)
    return desList

def encuentraClase(img,desList,thres=15):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList= []
    res = -1
    for d in desList:
        matches = bf.knnMatch(d,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        matchList.append(len(good))
    if len(matchList)!=0:
        if(max(matchList)>thres):
            res = matchList.index(max(matchList))
    return res

orb = cv2.ORB_create(nfeatures=1000)
path = 'imagenes_src'
nombres_imagenes = os.listdir(path)
lista_imagenes = []
nombresObjetos = []
for imagen in nombres_imagenes:
    img = cv2.imread(f'{path}/{imagen}',0)
    lista_imagenes.append(img)
    nombresObjetos.append(os.path.splitext(imagen)[0])
desList = descriptores(lista_imagenes)

path2 = 'imagenes'
imgComp = os.listdir(path2)
lista_img_checar = []
for i in imgComp:
    img2 = cv2.imread(f'{path2}/{i}')
    img2 = cv2.resize(img2, (600, 800)) 
    imgOr = img2.copy()
    clase = encuentraClase(img2,desList)
    if clase != -1:
        cv2.putText(imgOr,nombresObjetos[clase],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv2.imshow(f'{nombresObjetos[clase]} - {i}',imgOr)
    else:
        cv2.putText(imgOr,'N/A',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv2.imshow(f'N/A - {i}',imgOr)
cv2.waitKey(0) 
