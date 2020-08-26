import cv2

import facial_landmarks
from imutils import face_utils
import NN
import threading


#GLOBALS
frame = None
NAME = "unknown"
pred = None
detc = None
available = False
cache = False
rect = None

def verify_frame(pred, detc, le):
    global frame, NAME, available, cache, rect
    while True:
        if cache :
            state, shape, rect = facial_landmarks.test_frame(frame, pred, detc )
            if state:
                #FIND THE NAME
                _, embeddings = facial_landmarks.get_ratios(shape, frame)
                name, percentage = NN.predict_input_from_video(embeddings, le)
                percentage = float("{:.2f}".format(percentage))
                NAME = str(name) +" "+str(percentage)+ "%"
                available = True
            else:
                available = False
        cache = False
        

def main_loop(pred, detc):
    global frame, NAME, available, cache, rect
    #video path
    path_to_vid = '../video/george.mp4'
    video_capture = cv2.VideoCapture(path_to_vid)
    still = True
    frame_num = 0
    _, _, le = NN.prepare_data()
    verification_thread = threading.Thread(target=verify_frame, args=(pred, detc, le), daemon=True)
    verification_thread.start()
    while still:
        # Capture frame-by-frame
        still, frame_clear = video_capture.read()
        #if new, update frame
        if not cache:
            frame = frame_clear
            cache = True

        #DRAW ON THE FRAME
        if available:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame_clear, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_clear, NAME, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Video', frame_clear)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # print (frame_num)
        frame_num += 1

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


pred, detc = facial_landmarks.load_pred_detec()
main_loop(pred, detc)