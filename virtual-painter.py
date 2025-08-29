import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

lower_blue = np.array([100, 150, 70])
upper_blue = np.array([140, 255, 255])

canvas = None
prev_x, prev_y = None, None

brush_color = (255, 0, 0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pen_detected = False

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 50:
            pen_detected = True
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2

            
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), brush_color, 10)

           
            prev_x, prev_y = cx, cy

   
    if not pen_detected:
        prev_x, prev_y = None, None

    
    combined = canvas.copy()

    
    cv2.imshow("Virtual Painter", combined)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        print(" Canvas cleared!")

cap.release()
cv2.destroyAllWindows()




