import cv2 as cv

face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye = cv.CascadeClassifier("haarcascade_eye.xml")

cap = cv.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    faces = face.detectMultiScale(frame, 1.3, 10)

    for(x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        eyes = eye.detectMultiScale(frame, 1.1, 30)

        for(ex, ey, ew, eh) in eyes:

            cv.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
