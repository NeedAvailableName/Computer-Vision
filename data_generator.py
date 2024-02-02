import cv2
import os
label = "none"
start = False
cap = cv2.VideoCapture(0)

def count_image():
    if not os.path.exists('data/' + str(label)):
        return 0
    i = 0
    for file in os.listdir('data/' + label):
        i += 1
    return i
count = count_image()
i = 0
while(True):
    i += 1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow('frame',frame)
    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed == ord('s'):
        start = not start
    if start and i % 5 == 0:
        count += 1
        print("number image capture = ", count)
        if not os.path.exists('data/' + str(label)):
            os.mkdir('data/' + str(label))
        cv2.imwrite('data/' + str(label) + "/" + str(count) + ".png",frame)
        if count == 100: break
    if keyPressed == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
