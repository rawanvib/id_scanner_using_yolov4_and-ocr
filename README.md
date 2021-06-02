https://drive.google.com/file/d/1agIisEJJ9f2dXFoRTdCFphko8ZfKe3HQ/view?usp=sharing# id_scanner_using_yolov4_and-ocr
Extracted text using ocr techniques and created dataset out of it

##National_IDs_Images_Front_100-1

Training yolov4 for custom model and applying pytessearct to extract text

weights of our custom dataset : https://drive.google.com/file/d/1agIisEJJ9f2dXFoRTdCFphko8ZfKe3HQ/view?usp=sharing

IDE :- Jupyter Notebook

The Project is about gathering National IDs and extract the texts as per classes We used YOLOv4 to train our custom model so as to get bounding boxes at desired classes mentioned in obj.classes file To train the model we have to first annotate the data, for which we used labelimg application.

how to use labelimg :- https://youtu.be/p0nR2YsCY_U (check description for source code)

how to train custom yolov4 model :- https://youtu.be/mmj3nxGT2YQ

Now we gather the data and apply our custom yolov4 model get the bounding boxes and apply pytesseract in it and extract the texts out of images and save it in CSV

![Screenshot from 2021-06-01 16-13-48](https://user-images.githubusercontent.com/43780243/120421025-28f1cf00-c383-11eb-9154-ea50ad13184f.png)
![Screenshot from 2021-06-01 16-11-32](https://user-images.githubusercontent.com/43780243/120421027-2c855600-c383-11eb-8344-5c93f6ea2693.png)
