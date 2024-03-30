import cv2
import cvzone
from ultralytics import YOLO
import math


cap = cv2.VideoCapture('best.mp4')
#cap.set(3, 640)
#cap.set(4, 480)

weapon = YOLO('weapon.pt')
fire = YOLO('fireSomke.pt')
weaponClass=['Gun', 'Knife']
fireClass=['fire', 'smoke']
####### Opening webcam #######
while cap.isOpened():
    ret, frame = cap.read()

    weaponDetection = weapon(frame, stream=True)
    for r in weaponDetection:
        boxes = r.boxes
        for box in boxes:
            # Getting box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            '''
            x1, y1, x2, y2 contain the four cordinates of the square bounding box.
            To use those cordinates at first change them into integer 
            '''
            # Draw bounding box  using the cordinates -- using cvzone for fancy look
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(frame,(x1, y1, w, h))

            # Now drawing the bounding box using the coordinates -- using cv2
            #cv2.rectangle(img,(x1, y1), (x2, y2), (255,0,255),3)


            ### Showing the confidence level ####
            # Get the confidence on the box
            conf = math.ceil((box.conf[0]*100))/100
            # Show confidence on the box
            #cvzone.putTextRect(img,f'{conf}', (max(0,x1), max(35,y1)))
            '''
            To put text we used cvzone for better fit and view
            To take the text up use value(in this case 35) according to your need.
            '''

            ### Showing the class name ####
            cls = int(box.cls[0])
            cvzone.putTextRect(frame,f'{weaponClass[cls]}', (max(0,x1), max(35,y1)),scale=1, thickness=1)
            '''
            If don't want the confidence level then remove conf
            '''

    fireDetection = fire(frame, stream=True)
    for r in fireDetection:
        boxes = r.boxes
        for box in boxes:
            # Getting box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            '''
            x1, y1, x2, y2 contain the four cordinates of the square bounding box.
            To use those cordinates at first change them into integer 
            '''
            # Draw bounding box  using the cordinates -- using cvzone for fancy look
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))

            # Now drawing the bounding box using the coordinates -- using cv2
            # cv2.rectangle(img,(x1, y1), (x2, y2), (255,0,255),3)

            ### Showing the confidence level ####
            # Get the confidence on the box
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Show confidence on the box
            # cvzone.putTextRect(img,f'{conf}', (max(0,x1), max(35,y1)))
            '''
            To put text we used cvzone for better fit and view
            To take the text up use value(in this case 35) according to your need.
            '''

            ### Showing the class name ####
            cls = int(box.cls[0])
            cvzone.putTextRect(frame, f'{fireClass[cls]}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            '''
            If don't want the confidence level then remove conf
            '''

    if ret:
        vid_image = cv2.resize(frame, (600, 400))
        cv2.imshow("mid_vid", vid_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
