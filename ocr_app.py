import cv2
import numpy as np
import pytesseract
import os

per=25
pixelThreshold = 500

roi =  [[[(826,1052),(1118,1092)],'text','Candidate Name'],
        [[(826,1052),(1118,1092)],'text','Name of the Parent'],
        [[(826,1052),(1118,1092)],'text','Date of Birth'],
        [[(826,1052),(1118,1092)],'text','Signature']]
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\NEW\\tesseract-ocr-w64-setup-v5.0.0-alpha.20210506'

imgq = cv2.imread('img.jpg')
h,w,c = imgq.shape
imgq = cv2.resize(imgq,(w//4,h//4))

orb = cv2.ORB_create(1000)
kp1,dse1 = orb.detectAndCompute(imgq,None)
#impKp1 = cv2.drawKeypoints(imgq,kp1,None)

#cv2.imshow("KeyPointsQuery",impKp1)
path = 'UserForms'
myFormList = os.listdir(path)
print(myFormList)
for j,y in enumerate(myFormList):
    img = cv2.imread(path+"/"+y)
    img = cv2.resize(img, (w // 4, h // 4))
    kp2,dse2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(dse2,dse1)
    matches.sort(key=lambda x:x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img,kp2,imgq,kp1,good[:20],None,flags=2)
    #cv2.imshow(y,imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M,_ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
    imgScan = cv2.warpPerspective(img,M,(w,h))
    #imgScan = cv2.resize(imgScan, (w // 2, h // 2))
    #cv2.imshow(y, imgScan)

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)
    myData = []
    print(f'Extracting data from form {j}')
    for x,r in enumerate(roi):
        cv2.rectangle(imgMask,(r[0][0], r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)

        imgCrop = imgScan[r[0][1]:r[1][1],r[0][0]:r[1][0]]
        #cv2.imshow(str(x), imgCrop)

        if r[2] == 'text':
           print('{} :{}'.format(r[3],pytesseract.image_to_string(imgCrop)))
           myData.append(pytesseract.image_to_string(imgCrop))
        if r[2] == 'box':
            imgGrey = cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgGrey,170,255,cv2.THRESH_BINARY.INV)[1]
            totalPixels = cv2.countNonZero(imgThresh)
            if totalPixels>pixelThreshold: totalPixels =1
            else: totalPixels=0
            print(f'{r[3]}:{totalPixels}')
            myData.append(totalPixels)
        cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),
                    cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)
    with open('DataOutput.csv','a+') as f:
        for data in myData:
            f.write((str(data)+','))
        f.write('\n')
    imgShow = cv2.resize(imgShow, (w // 2, h // 2))
    print(myData)
    cv2.imshow(y+"2",imgShow)


#cv2.imshow("Output",imgq)
cv2.waitKey(0)
