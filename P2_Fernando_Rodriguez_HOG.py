import cv2
import numpy as np
import os

winStride = (8,8)
padding = (8,8)
locations = ((10,20),)


def descriptores(images):
    desList=[]
    for i in images:
        des = hog.compute(i,winStride,padding,locations)
        desList.append(des)
    return desList

def encuentraClase(img,desList,thres=15):
    des2 = hog.compute(img,winStride,padding,locations)
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

winSize = (60,60)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
##No modificar
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
##
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride, 
cellSize,nbins,derivAperture,winSigma,histogramNormType,
L2HysThreshold,gammaCorrection,nlevels)


path = 'imagenes_src'
nombres_imagenes = os.listdir(path)
lista_imagenes = []
nombresObjetos = []
for imagen in nombres_imagenes:
    img = cv2.imread(f'{path}/{imagen}',0)
    img = cv2.resize(img, (64, 128))
    lista_imagenes.append(img)
    nombresObjetos.append(os.path.splitext(imagen)[0])
desList = descriptores(lista_imagenes)

path2 = 'imagenes'
imgComp = os.listdir(path2)
lista_img_checar = []
for i in imgComp:
    img2 = cv2.imread(f'{path2}/{i}')
    img2 = cv2.resize(img2, (300,400)) 
    imgOr = img2.copy()
    clase = encuentraClase(img2,desList)
    if clase != -1:
        cv2.putText(imgOr,nombresObjetos[clase],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv2.imwrite(f'{nombresObjetos[clase]} - {i}',imgOr)
    else:
        cv2.putText(imgOr,'N/A',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv2.imwrite(f'N/A - {i}',imgOr)
cv2.waitKey(0) 
