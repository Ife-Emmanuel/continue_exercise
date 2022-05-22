import numpy as np
import os
import cv2
import json
from users.users_details import users_profile
#first extract all regular image files in case there folders present so as not to prompt error.


for user in users_profile.keys():
    path = users_profile[user]['captures']
    files = [image for image in os.listdir(path) if os.path.isfile(f"{path}/{image}")]
    inputs = []
    targets = []

    for i, file in enumerate(files):
        image_path = f"{path}/{file}"
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        inputs.append(np.asarray(image, dtype= np.uint8))
        targets.append(i)
        # image = np.asarray(image, dtype= np.uint8)
        inputs.append(np.asarray(image, dtype= np.uint8))
        targets.append(i)

    targets = np.asarray(targets, dtype= np.int32)

    classifier = cv2.face.LBPHFaceRecognizer_create()
    # Train classifier
    classifier.train(np.asarray(inputs), np.asarray(targets))
    users_profile[user]['classifier'] = classifier

print(users_profile)

#
# path = 'users.json'
# with open(path, 'w') as fp:
#     json.dump(users_profile, fp, indent=4)