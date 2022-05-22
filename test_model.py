import cv2
import numpy as np
import trainModel

cascade_path = r'C:\Users\User\PycharmProjects\computer_vision_projects\face_recognition\cascades\haarcascade_frontalface_default.txt'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Find face and return the original image and cropped face

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if faces is None:
        return image

    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
            # cv2.rectangle(image, (x, y) (x + w, y + h), (0, 255, 0), 2)
            cropped_face = image[ y : y + h, x : x + w]
            resized_face = cv2.resize(cropped_face, (200, 200))
            return resized_face


cap = cv2.VideoCapture(0)
_, frame1 = cap.read()


while True:
    found_face = detect_face(frame1)
    h, w, c = frame1.shape
    text_position = (int(w/2) - 50, int(h/2))

    try:
        gray_face = cv2.cvtColor(found_face, cv2.COLOR_BGR2GRAY)

        # predict using trained model
        label, score = trainModel.classifier.predict(gray_face)
        if score < 500:
            confidence_score = int(100 * (1 - score/400))
            message = 'I am {} confident '.format(str(confidence_score))
        cv2.putText(frame1, message, text_position, cv2.FONT_ITALIC, 1, (243, 0, 243))

        if score > 75:
            message = 'Sorry! You are not who you say'
            cv2.putText(frame1, message, text_position, cv2.FONT_ITALIC, 1, (200, 0, 200), 2)
            cv2.imshow('Recognition', frame1)

        else:
            message = 'welcome. you are finally here.'
            pre_texted_frame1 = frame1
            cv2.putText(frame1, message, text_position, cv2.FONT_ITALIC, 1, (0, 200, 200), 2)
            cv2.imshow('Recognition', frame1)

    except:
        message = 'No face found!'
        cv2.putText(frame1, message, text_position, cv2.FONT_ITALIC, 1, (0, 200, 200), 2)
        cv2.imshow('Recognition', frame1)
    _, frame2 = cap.read()
    frame1 = frame2

    if cv2.waitKey(1) == 113:
        break

cap.release()
cv2.destroyAllWindows()