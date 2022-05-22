"""To generate data needed for training"""
import cv2
import numpy as np
import os
import shutil
from users.users_details import users_profile

face_classifier_path = r'C:\Users\User\PycharmProjects\computer_vision_projects\face_recognition\cascades\haarcascade_frontalface_default.txt'
face_cascade = cv2.CascadeClassifier(face_classifier_path)

def get_face(frame):
    """to get cropped face and it location"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if not len(faces):
        return None
    elif faces.any():
        for (x, y, w, h) in faces:
            cropped_face = frame[y: y + h, x: x + w]
        return (x, y, w, h), cropped_face



def generate_data(user_data_folder):
    cap = cv2.VideoCapture(0)
    num_faces = 0
    new_dimension = (200, 200)
    i = 1
    cv2.namedWindow('saving training data')

    while cap.isOpened():
        _, frame = cap.read()
        if get_face(frame):
            num_faces += 1

            (x, y, w, h), found_face = get_face(frame)
            # resizing found face to new dimensions
            resized_face = cv2.resize(found_face, new_dimension)
            resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)

            # Saving the training data
            capture_no = str(num_faces) + '.jpg'
            path = os.path.join(user_data_folder, capture_no)
            cv2.imwrite(path, resized_face)

            # To track number of images completed
            cv2.putText(resized_face, str(num_faces), (50, 50), cv2.FONT_ITALIC, 1, (0, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 250, 0), 2)
            cv2.imshow('Saving training data', frame)
            cv2.imshow('found face', resized_face)

        else:
            frame = np.zeros_like(frame)
            text1 = ' Saving Error : '
            text2 = ' No face detected.'
            cv2.putText(frame, text1, (50, 100), cv2.FONT_ITALIC, 2, (0, 0, 255), 4)
            cv2.putText(frame, text2, (50, 150), cv2.FONT_ITALIC, 2, (0, 0, 255), 4)
            cv2.imshow('Saving training data', frame)

        if cv2.waitKey(2) == 113 or num_faces == 50:
            break
            i += 1
    cap.release()


cwd = os.getcwd()

def prompt_data_generations():
    while True:
        name = input('Enter your name : ')
        # Allowing the name of the user to start the name of his/her training data directory
        user_data_folder = os.path.join(cwd, f"training_data/{name.capitalize()}_captures")
        if name in users_profile.keys():
            i = 1
            while True:
                response = input(f"{name.capitalize()} already exists. If its your account, are you ok with available data?(y/n)")
                #** you don't just allow the user to regenerate if response is yes.. you can run test model to detect face before allowing regeneration of data.
                if response == 'n':
                    new_account_response = input(f"{name.capitalize()} do you wan't to change your id or username?(y/n)")
                    if new_account_response == 'y':
                        new_name = input('Enter a new name or id : ')
                        gender = input('Enter your gender? ')
                        users_profile[new_name] = users_profile.pop(name.lower())
                        shutil.rmtree(user_data_folder)
                        users_profile[new_name]['gender'] = gender
                        user_data_folder = os.path.join(cwd, f"training_data/{name.capitalize()}_captures")
                        os.mkdir(user_data_folder)
                    generate_data(user_data_folder)

                break
        else:
            gender = input('What is your gender(male/female) : ')

            #preparing the user for a set of captures
            active = True
            while active:
                answer = input('Another capture about to start. If ready press \'y\' : ')
                if answer == 'y':
                    active = False

            users_profile[name] = {'name': name.lower(), 'gender': gender}
            os.mkdir(user_data_folder)
            generate_data(user_data_folder)
        second_response = input('Are the required users completed? ')
        if second_response == 'y':
                break
        continue

    cv2.destroyAllWindows()

prompt_data_generations()
print(users_profile)