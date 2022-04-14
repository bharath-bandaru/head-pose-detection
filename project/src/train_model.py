import cv2 as cv
from PIL import Image


images_path = './data/widerface/train/images/'
label_txt = images_path[:-7] + 'label.txt'
index = 0
with open(label_txt, 'r') as fr:
    test_dataset = fr.read().split("\n")
test_dataset

images = []
pose_labels = []
for i, img_name in enumerate(test_dataset):
    if(img_name[0] == '#'):
        images.append(Image.open(images_path+img_name[2:]))
        pose_labels