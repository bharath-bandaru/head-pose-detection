import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import dlib
from imutils import face_utils
import keras
import os
import numpy as np
import cv2
import scipy.io as sio
from math import cos, sin
import dlib

INPUT_SIZE = 240
alpha=0.1
def loss_function(y_true, y_pred):
    real_true = y_true[:, 0]
    bin_true = y_true[:, 1]
    cat_f = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True)
    bin_one_hot = tf.keras.utils.to_categorical(bin_true.numpy(), 66)
    cls_loss = cat_f(bin_one_hot, y_pred)
    # # MSE loss
    pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * idx_tensor, 1) * 3 - 99
    mse_loss = tf.losses.mean_squared_error(real_true, pred_cont)
    # # Total loss
    total_loss = cls_loss + alpha * mse_loss
    return total_loss
print("model loading")
model = keras.models.load_model('../data/models/model1.h5',custom_objects={"loss_func": loss_function})
print("model loaded")
cap = cv2.VideoCapture(0)
print("camera loading")
if not cap.isOpened():
    print("Unable to connect to camera.")
    exit(-1)
face_landmark_path = '../data/models/shape_predictor_68_face_landmarks.dat'
idx_tensor = [idx for idx in range(66)]
idx_tensor = tf.Variable(np.array(idx_tensor, dtype=np.float32))

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
def crop_face_loosely(shape, img, input_size):
    x = []
    y = []
    for (_x, _y) in shape:
        x.append(_x)
        y.append(_y)

    max_x = min(max(x), img.shape[1])
    min_x = max(min(x), 0)
    max_y = min(max(y), img.shape[0])
    min_y = max(min(y), 0)

    Lx = max_x - min_x
    Ly = max_y - min_y

    Lmax = int(max(Lx, Ly) * 2.0)

    delta = Lmax // 2

    center_x = (max(x) + min(x)) // 2
    center_y = (max(y) + min(y)) // 2
    start_x = int(center_x - delta)
    start_y = int(center_y - delta - 30)
    end_x = int(center_x + delta)
    end_y = int(center_y + delta - 30)

    if start_y < 0:
        start_y = 0
    if start_x < 0:
        start_x = 0
    if end_x > img.shape[1]:
        end_x = img.shape[1]
    if end_y > img.shape[0]:
        end_y = img.shape[0]

    crop_face = img[start_y:end_y, start_x:end_x]

    cv2.imshow('crop_face', crop_face)

    crop_face = cv2.resize(crop_face, (input_size, input_size))
    input_img = np.asarray(crop_face, dtype=np.float32)
    normed_img = (input_img - input_img.mean()) / input_img.std()

    return normed_img
bins = np.array(range(-99, 102, 3))
def test_model(frames):
    val =model.predict(np.array(frames))
    pred_cont_yaw = bins[np.argmax(val[0])]
    pred_cont_pitch = bins[np.argmax(val[1])]
    pred_cont_roll = bins[np.argmax(val[2])]
    return pred_cont_yaw,pred_cont_pitch,pred_cont_roll
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        face_rects = detector(frame, 0)
        if len(face_rects) > 0:
            shape = predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)
            face_crop = crop_face_loosely(shape, frame, INPUT_SIZE)
            frames.append(face_crop)
            if len(frames) == 1:
                print(shape[30])
                pred_cont_yaw, pred_cont_pitch, pred_cont_roll = test_model(np.array(frames))
                cv2_img = draw_axis(frame, pred_cont_yaw, pred_cont_pitch, pred_cont_roll, tdx=shape[30][0],
                                          tdy=shape[30][1], size=100)
                cv2.imshow("cv2_img", cv2_img)
                frames = []
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break