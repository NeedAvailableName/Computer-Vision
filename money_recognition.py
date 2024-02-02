import numpy as np
import cv2
from util.my_model import get_model, rescale, get_labels
from collections import deque
# get 3 frame to predict
Q = deque(maxlen=3)
# lower bound of prediction result
threshold = 0.8
my_model = get_model()
my_model.load_weights("weight\\final_weights.hdf5")
screen_width = 1920  # Replace with the width of your display
screen_height = 1080
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
class_name = get_labels()
while True:
    ret, image_org = cap.read()
    if not ret:
        continue
    # resize to 80% of original image
    image_org = cv2.resize(image_org, dsize=None, fx=0.8, fy=0.8)
    image = image_org.copy()
    image = cv2.resize(image, dsize=(224, 224))
    image = np.expand_dims(image, axis=0)
    image = rescale(image)
    predict = my_model.predict(image)
    Q.append(predict[0])
    mean_results = np.array(Q).mean(axis=0)
    print("The value is: ", class_name[np.argmax(predict[0])], (predict[0]))
    print(np.max(predict[0],axis=0))
    if (np.max(mean_results) >= threshold) and (np.argmax(mean_results) != 8):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (0, 0, 255)
        thickness = 2
        cv2.putText(image_org, class_name[np.argmax(mean_results)], org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Currency Recognition", image_org)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
