import cv2
import os
import imutils
import time

mano = 'piedra'
#mano = 'papel'
#mano = 'tijeras'
#mano = 'arriba'
#mano = 'abajo'

dataPath = r'C:\Users\HP\Desktop\Machine Learning\piedra_papel_tijeras'

Path = dataPath + '/' + mano

if not os.path.exists(Path):
  print('Carpeta creada: ',Path)
  os.makedirs(Path)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

i = 0

while True:

  ret, frame = cap.read()
  if ret == False:
           break
        
  frame =  imutils.resize(frame, width=640)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  auxFrame = frame.copy()
  cv2.rectangle(frame, (50,100),(250,300),(0,255,0),2)
  roi = auxFrame[100:300,50:250]

  font = cv2.FONT_HERSHEY_SIMPLEX
  
  cv2.putText(frame, "Collecting {}".format(i),(5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
        
    roi = cv2.resize(roi,(150,150),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(Path + '/img_{}.jpg'.format(i),roi)

    i+=1
    

        
            
  cv2.imshow('roi',roi)
  cv2.imshow('frame',frame)

  if (cv2.waitKey(1) & 0xFF == ord('e')) or (i==200):
    break

cap.release()
cv2.destroyAllWindows()
