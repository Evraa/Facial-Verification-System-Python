import cv2

import facial_landmarks
from imutils import face_utils
import NN


def main_loop(pred, detc):

    #video path
    path_to_vid = '../video/george.mp4'
    video_capture = cv2.VideoCapture(path_to_vid)
    still = True
    frame_num = 0
    rect_last = None
    _, _, le = NN.prepare_data()
    NAME = 'unknown'
    while still:
        # Capture frame-by-frame
        still, frame = video_capture.read()
        # Display the resulting frame
        if frame_num%30 == 0:
            state, shape, rect = facial_landmarks.test_frame(frame, pred, detc )
            if state:
                rect_last = rect
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #FIND THE NAME
                _, embeddings = facial_landmarks.get_ratios(shape, frame)
                name, percentage = NN.predict_input_from_video(embeddings, le)
                percentage = float("{:.2f}".format(percentage))
                NAME = str(name) +" "+str(percentage)+ "%"
                if name != "George_W_Bush":
                    cv2.putText(frame, NAME, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # show the face number
                else:
                    cv2.putText(frame, NAME, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif rect_last != None:
            (x, y, w, h) = face_utils.rect_to_bb(rect_last)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the face number
            if name != "George_W_Bush":
                cv2.putText(frame, NAME, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(frame, NAME, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        cv2.imshow('Video', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # print (frame_num)
        frame_num += 1

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


pred, detc = facial_landmarks.load_pred_detec()
main_loop(pred, detc)