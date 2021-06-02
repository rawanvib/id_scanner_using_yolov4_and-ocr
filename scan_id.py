import os
import cv2
import math
import easyocr
import pytesseract
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


# searches for images in a folder and returns a list containing path of the image
def load_images_from_folder(folder):
    images = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or filename.endswith('.JPG') \
                or file_name.endswith('.JPEG') or file_name.endswith('.png') or filename.endswith('.PNG'):
            img = os.path.join(folder, file_name)

            if img is not None:
                images.append(img)
    return images


# takes a list of images, returns a list containing images that are rotated with a specific angle if found or else do not rotate it
def rotated_images(list_of_images):
    rotated_images = []
    for img in tqdm(list_of_images):

        img = cv2.imread(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray_gaussian = cv2.GaussianBlur(img_gray, (3, 3), 0)

        img_edges = cv2.Canny(gray_gaussian, 50, 150)

        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

        # if no angle found return the original image
        if lines is None:
            median_angle = 0.0
        else:
            angles = []
            for [[x1, y1, x2, y2]] in lines:
                # convert radian into degrees
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)
                median_angle = np.median(angles)

        if median_angle == -90.0:
            img_rotated = img
        else:
            img_rotated = ndimage.rotate(img, median_angle)

        rotated_images.append(img_rotated)
    return rotated_images


# takes list of images and apply ocr on detected objects
def extract_text_from_id_using_pytesseract(list_rotated_images):
    data_from_all_ids = []
    score = 0
    denominator = 0
    for img_rotated in tqdm(list_rotated_images):

        text_from_id = []
        image = img_rotated.copy()
        classes, scores, boxes = model.detect(img_rotated, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        list_label = ['nameline1', 'nameline2', 'addressline1', 'addressline2', 'UID']
        for (box, classid) in zip(boxes, classes):
            if classNames[classid[0]] in list_label:
                # label = classNames[classid[0]]
                x, y, l, b = box
                cropped = image[y:b + y, x:x + l]
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_data(cropped, output_type=pytesseract.Output.DICT, lang='ara+en')

                inside_list = []
                for i in range(0, len(text['text'])):
                    inside_list.append(text['text'][i])
                    if text['conf'][i] == '-1':
                        continue
                    else:
                        denominator += 100
                        score += text['conf'][i]

                text_from_id.append(inside_list)
        data = dict(zip(list_label, text_from_id))
        data_from_all_ids.append(data)
    percent = (score / denominator) * 100
    print(f"accuracy of pytesseract {percent}")
    return data_from_all_ids, percent


def extract_text_from_id_using_easyOcr(list_rotated_images):
    data_from_all_ids = []
    score = 0
    denominator = 0
    for img_rotated in tqdm(list_rotated_images):

        text_from_id = []
        image = img_rotated.copy()
        classes, scores, boxes = model.detect(img_rotated, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        list_label = ['nameline1', 'nameline2', 'addressline1', 'addressline2', 'UID']
        for (box, classid) in zip(boxes, classes):
            if classNames[classid[0]] in list_label:
                # label = classNames[classid[0]]
                x, y, l, b = box
                cropped = image[y:b + y, x:x + l]
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                text = reader.readtext(cropped)

                inside_list = []
                for i in range(0, len(text)):
                    inside_list.append(text[i][1])
                    if text[i][2] == 0.0:
                        continue
                    else:
                        denominator += 1
                        score += text[i][2]

                text_from_id.append(inside_list)
        data = dict(zip(list_label, text_from_id))
        data_from_all_ids.append(data)
    percent = (score / denominator) * 100
    print(f"accuracy of easyocr {percent}")
    return data_from_all_ids, percent


if __name__ == '__main__':
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4

    classFile = './obj.names'
    classNames = []

    with open(classFile, 'rt') as f:
        classNames = f.read().splitlines()
    print(classNames)

    configPath = './yolov4-obj.cfg'
    weightsPath = './yolov4-obj_last.weights'

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    reader= easyocr.Reader(['en', 'ar'])

    folder_path = "/home/webwerks/Desktop/ocr/new_id_ocr/test"
    path_of_images = load_images_from_folder(folder_path)
    new_images = rotated_images(path_of_images)
    list_of_data_using_pytesseract, percent_of_pytesseract = extract_text_from_id_using_pytesseract(new_images)
    list_of_data_using_easyOcr, percent_of_easyOcr = extract_text_from_id_using_easyOcr(new_images)

    # checking accuracy of easyocr and pytesseract , creating dataframe using that ocr software which gives higher accuracy
    if percent_of_easyOcr>percent_of_pytesseract:
        data=pd.DataFrame(list_of_data_using_easyOcr)
    else:
        data=pd.DataFrame(list_of_data_using_pytesseract)


    print(data)
    data.to_csv('id_data1.csv')
