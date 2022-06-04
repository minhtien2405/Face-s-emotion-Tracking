import cv2
import os
from deepface import DeepFace


def emotion_tracking(my_path):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(my_path)

    try:
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')

    # creating value model training
    value = {
        'angry':0,
        'disgust':0,
        'fear':0,
        'happy':0,
        'sad':0,
        'surprise':0,
        'neutral':0
    }

    # frame
    while True:
        ret,frame = cap.read()
    #     frame = cv2.flip(frame,1)
        if ret:
            try:
                result = DeepFace.analyze(frame,actions = ['emotion'])
                if result['dominant_emotion'] in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                    value[result['dominant_emotion']]+=1
            except:
                continue
        else:
            break
    # print(value)

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    new_value = dict(sorted(value.items(), key=lambda item: item[1], reverse= True))
    check_value_key = list(new_value.keys())
    if check_value_key[0] == 'neutral':
        if new_value[list(new_value.keys())[1]] != 0:
            return check_value_key[1]
        else:
            return 'newtral'
