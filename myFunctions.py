import numpy as np
import mediapipe as mp
import mediapipeFuncions
import cv2
import os
from keras.models import load_model
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils          # 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # 繪圖樣式
mp_pose = mp.solutions.pose                      # 姿勢偵測
# Functions defined to detect Level1 and Level2 data
answer = np.array([])

def detectUpAndDown(result_arr):
    global answer
    init_val = result_arr[0]
    final_val = result_arr[-1]
    diff = init_val - final_val
#     print("detectUpAndDown:")
#     print(init_val, final_val)
    if(diff <= -0.1):
        print('\033[1;35mAns:Sit down \033[0m')
        answer = np.append(answer, 2)
        return True
    elif(diff > 0.1):
        print('\033[1;35mAns:Stand up \033[0m')
        answer = np.append(answer, 3)
        return True
    
def detectHandWave(result_arr):
    max_val = result_arr[np.argmax(result_arr)]
    min_val = result_arr[np.argmin(result_arr)]
    diff = max_val - min_val
#     print("detectHandWave:")
#     print(max_val, min_val)
    if(diff > 0.05):
        return True
    
def detectHandWave_hands(right_arr, left_arr, right_arr_y, left_arr_y):
    global answer
    max_val_r = right_arr_y[np.argmax(right_arr_y)]
    min_val_r = right_arr_y[np.argmin(right_arr_y)]
    max_val_l = left_arr_y[np.argmax(left_arr_y)]
    min_val_l = left_arr_y[np.argmin(left_arr_y)]
   
    diff_r = max_val_r - min_val_l
    dfff_l = max_val_l - min_val_r
#     print("hands_y", diff_r, dfff_l)
    if(diff_r > 0.2 or dfff_l > 0.2):
        if(detectHandWave(right_arr) or detectHandWave(left_arr)):
            print('\033[1;31mAns:Hand waving \033[0m')
            answer = np.append(answer, 0)
            return True
        
def detectKicking(result_arr):
    max_val = result_arr[np.argmax(result_arr)]
    min_val = result_arr[np.argmin(result_arr)]
    diff = max_val - min_val
#     print("detectKicking:")
#     print(max_val, min_val)
    if(diff > 0.13):
        return True
    
def detectKicking_feet(right_arr, left_arr, right_arr_z, left_arr_z):
    global answer
    if(detectKicking(right_arr) or detectKicking(left_arr) or detectKicking(right_arr_z) or detectKicking(left_arr_z)):
        print('\033[1;31mAns:Kicking something\033[0m')
        answer = np.append(answer, 1)
        return True
    
def is_leve1Orlevel2(cap):
    right_eye = np.array([]) 
    left_eye = np.array([]) 
    right_thumb = np.array([])
    left_thumb = np.array([])
    right_wrist = np.array([])
    left_wrist = np.array([])
    right_ankle = np.array([])
    left_ankle = np.array([])
    right_ankle_z = np.array([])
    left_ankle_z = np.array([])
    i = 0
    # 啟用姿勢偵測
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            img_cut = img[135:850, 640:1440]

    #         img_cut_binary = binary(img_cut)

    #       img = cv2.resize(img,(520,300))               # 縮小尺寸，加快演算速度
            img2 = cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
            results = pose.process(img2)                  # 取得姿勢偵測結果
            # 根據姿勢偵測結果，標記身體節點和骨架
            landmarks = results.pose_landmarks.landmark
    #         print(landmarks[mp_pose.PoseLandmark["LEFT_WRIST"].value].x)

            right_eye = np.append(right_eye, landmarks[mp_pose.PoseLandmark["RIGHT_EYE"].value].y)

            right_thumb = np.append(right_thumb, landmarks[mp_pose.PoseLandmark["RIGHT_THUMB"].value].x)
            left_thumb = np.append(left_thumb, landmarks[mp_pose.PoseLandmark["LEFT_THUMB"].value].x)
            right_thumb_y = np.append(right_thumb, landmarks[mp_pose.PoseLandmark["RIGHT_THUMB"].value].y)
            left_thumb_y = np.append(left_thumb, landmarks[mp_pose.PoseLandmark["LEFT_THUMB"].value].y)

            
            right_wrist = np.append(right_wrist, landmarks[mp_pose.PoseLandmark["RIGHT_WRIST"].value].x)
            left_wrist = np.append(left_wrist , landmarks[mp_pose.PoseLandmark["LEFT_WRIST"].value].x)

            right_ankle = np.append(right_ankle , landmarks[mp_pose.PoseLandmark["RIGHT_ANKLE"].value].x)
            left_ankle = np.append(left_ankle , landmarks[mp_pose.PoseLandmark["LEFT_ANKLE"].value].x)
            right_ankle_z = np.append(right_ankle , landmarks[mp_pose.PoseLandmark["RIGHT_ANKLE"].value].z)
            left_ankle_z = np.append(left_ankle , landmarks[mp_pose.PoseLandmark["LEFT_ANKLE"].value].z)
            
            mp_drawing.draw_landmarks(
                img_cut,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('output', img_cut)

            i = i + 1
            if cv2.waitKey(5) == ord('q'):
                break     # press q to stop
    cap.release()
    cv2.destroyAllWindows()
    
    right_ankle_z  = np.delete(right_ankle_z, -1)
    left_ankle_z = np.delete(left_ankle_z, -1)
    is_detectUpAndDown = detectUpAndDown(right_eye)

   
    if(not(is_detectUpAndDown)): 
        is_detect_Kicking = detectKicking_feet(right_ankle, left_ankle, right_ankle_z, left_ankle_z)
        if(not(is_detect_Kicking)):
            is_detectHandWave = detectHandWave_hands(right_thumb, left_thumb, right_thumb_y, left_thumb_y)
    if(not(is_detectUpAndDown) and not(is_detect_Kicking) and not(is_detectHandWave) ):
        print("not level 1 or level 2, use the model to prediect data")
        return True

#-------------------------------------------------------------------------------------------------

# Functions defined to detect Level3 data

def detectLevelThree(video_test):
    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('test_level_three') 
#     try:
#         os.remove(DATA_PATH)
    

    # Actions that we try to detect
#     actions = np.array(['Reading', 'Writing', 'PlayWithPhone'])
#     actions = np.array([])
    # Thirty videos worth of data
#     no_sequences = 2

    # Videos are going to be 60 frames in length
    sequence_length = 60

    # Folder start
    start_folder = 0

 
        
    # Level3
    # Reading 
#     video_reading_test  = np.array([
#                                'C:/Users/Josh/testing/3/S001C001P006R002A011_rgb.avi',
#                                'C:/Users/Josh/testing/3/S001C001P007R001A011_rgb.avi'])
#     # Writing
#     video_writing_test  = np.array([
#                                'C:/Users/Josh/testing/3/S001C001P006R002A012_rgb.avi',
#                                'C:/Users/Josh/testing/3/S001C001P007R001A012_rgb.avi'])
#     # Play with phone
#     video_play_test = np.array([
#                                'C:/Users/Josh/testing/3/S001C001P006R002A029_rgb.avi',
#                                'C:/Users/Josh/testing/3/S001C001P007R001A029_rgb.avi'])

    # Set mediapipe model 
    
    for idx, addr in enumerate(video_test):
        try:
            os.makedirs(os.path.join(DATA_PATH, str(idx)))
            print(os.path.join(DATA_PATH, str(idx)))
        except:
            pass
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for idx, addr in enumerate(video_test):
            cap = cv2.VideoCapture(addr)
        
            for frame_num in range(sequence_length):
            # Read feed

                ret, frame = cap.read()
                frame_cut = frame[135:850, 640:1440]

                # Make detections
                image, results = mediapipeFuncions.mediapipe_detection(frame, holistic)

                # Draw landmarks
#                 draw_styled_landmarks(image, results)
    #                     print(frame_num)
                # NEW Apply wait logic
                if frame_num == 0: 

#                     cv2.putText(image, 'STARTING COLLECTION', (120,200), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
#                     cv2.waitKey(10)
                else: 
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                # NEW Export keypoints
                keypoints = mediapipeFuncions.extract_keypoints(results)
#                 print(idx)
                npy_path = os.path.join(DATA_PATH, str(idx), str(frame_num))
#                 print(npy_path)
                np.save(npy_path, keypoints)



                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q') or frame_num == 59:
                    break
    cap.release()
    cv2.destroyAllWindows()
#     global model
#     del model
    
    
#     labels = [0,0,1,1,2,2]
    X_test =([])
    sequences = []
    for idx, addr in enumerate(video_test):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, str(idx), "{}.npy".format(frame_num)))
            
            window.append(res)
        sequences.append(window)
    X_test = np.array(sequences)
#     print(X_test.shape)
#     y_test = to_categorical(labels).astype(int)
    model = load_model('action.h5')
    model.summary()
    yhat = model.predict(X_test)
#     print(yhat)
#     ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    #(0 :'Reading', 1:'Writing', 2:'PlayWithPhone'])
#     print(ytrue,yhat)
#     print('Level3',accuracy_score(ytrue, yhat))
    global answer
    for ans in yhat:
        answer = np.append(answer, ans+4)
    
#     DATA_PATH = os.path.join('MP_Data_test')
def getAnswer():
    global answer
    return answer

def resetAnswer():
    global answer 
    answer = np.array([]) 