import cv2
import os
import imutils
import time

mano = ['piedra', 'papel', 'tijeras', 'arriba','abajo', 'nada']
directory = os.getcwd()
os.makedirs(directory + '/imgs', exist_ok=True)
path = directory + '/imgs'

for m in mano:
  os.makedirs(path + '/' + m, exist_ok=True)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

for m in mano:
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
    font1 = cv2.FONT_ITALIC

    cv2.putText(frame, "Capturando {} {}".format(m, i),(5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Presione n para cambiar posicion a capturar",(5, 380), font1, 0.5, (187, 47, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Presione q para capturar la imagen",(5, 400), font1, 0.5, (187, 47, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Presione e para salir",(5, 420), font1, 0.5, (187, 47, 0), 1, cv2.LINE_AA)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        
      roi = cv2.resize(roi,(180,180),interpolation=cv2.INTER_CUBIC)
      cv2.imwrite(path+'/'+ m + '/img_{}.jpg'.format(i),roi)

      i+=1
                
    cv2.imshow('Captura de imagenes',frame)

    if (cv2.waitKey(1) & 0xFF == ord('n')) or (i==300):
      break

    if (cv2.waitKey(1) & 0xFF == ord('e')):
      cap.release()
      cv2.destroyAllWindows()
  time.sleep(2)
    
cap.release()
cv2.destroyAllWindows()
