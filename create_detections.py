"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm
import argparse
from det_util import generate_detections
import pandas
import cv2
import os

"""
Inference script to generate a file of predictions given an input.

EXAMPLE
python create_detections.py 104.tif -c model_Octonauts.pb -score 0.5
python create_detections.py 104.tif -c model_4246k.pb -score 0.5

Args:
    checkpoint: A filepath to the exported pb (model) file.
        ie ("saved_model.pb")

    chip_size: An integer describing how large chips of test image should be

    input: A filepath to a single test chip
        ie ("1192.tif")

    score_prediction: A float describing at greater than or equal to what score should we include boxes

    output: A filepath where the script will save  its predictions
        ie ("predictions.txt")


Outputs:
    Writes a file specified by the 'output' parameter containing predictions for the model.
        Per-line format:  xmin ymin xmax ymax class_prediction score_prediction
        Note that the variable "num_preds" is dependent on the trained model 
        (default is 250, but other models have differing numbers of predictions)

"""

def chip_image(img, chip_size=(300,300)):
    """
    Segment an image into NxWxH chips

    Args:
        img : Array of image to be chipped
        chip_size : A list of (width,height) dimensions for chips

    Outputs:
        An ndarray of shape (N,W,H,3) where N is the number of chips,
            W is the width per chip, and H is the height per chip.

    """
    width,height,_ = img.shape 
    wn,hn = chip_size
    images = np.zeros((int(width/wn) * int(height/hn),wn,hn,3))   
    k = 0
    for i in tqdm(range(int(width/wn))):
        for j in range(int(height/hn)):
            
            chip = img[wn*i:wn*(i+1),hn*j:hn*(j+1),:3]  
            images[k]=chip
            
            k = k + 1
    
    return images.astype(np.uint8)

def detect(checkpoint, input):

    score_prediction = 0.5
    chip_size = 300
    

    #Histogram Equalize the image
    img = cv2.imread(input, 0)
    histImg = cv2.equalizeHist(img)
    clahe3 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl13 = clahe3.apply(histImg)

    #Parse and chip images
    arr = np.array(np.expand_dims(cl13, axis=2))       
    chip_size = (chip_size,chip_size)
    images = chip_image(arr,chip_size)

    #generate detections
    boxes, scores, classes = generate_detections(checkpoint,images)

    #Process boxes to be full-sized
    width,height,_ = arr.shape
    cwn,chn = (chip_size)
    wn,hn = (int(width/cwn),int(height/chn))

    num_preds = 250
    bfull = boxes[:wn*hn].reshape((wn,hn,num_preds,4))
    b2 = np.zeros(bfull.shape)
    b2[:,:,:,0] = bfull[:,:,:,1]
    b2[:,:,:,1] = bfull[:,:,:,0]
    b2[:,:,:,2] = bfull[:,:,:,3]
    b2[:,:,:,3] = bfull[:,:,:,2]

    bfull = b2
    bfull[:,:,:,0] *= cwn
    bfull[:,:,:,2] *= cwn
    bfull[:,:,:,1] *= chn
    bfull[:,:,:,3] *= chn
    for i in range(wn):
        for j in range(hn):
            bfull[i,j,:,0] += j*cwn
            bfull[i,j,:,2] += j*cwn
            
            bfull[i,j,:,1] += i*chn
            bfull[i,j,:,3] += i*chn
            
    bfull = bfull.reshape((hn*wn,num_preds,4))

    #get the file names
    img_name = os.path.basename(input).split('.')[0]
    model_name = os.path.basename(checkpoint).split('_')[1].split('.')[0]
    file_name = img_name + '_' + model_name + '_preds.txt'
    with open(file_name,'w') as f:
        for i in range(bfull.shape[0]):
            for j in range(bfull[i].shape[0]):
                #box should be xmin ymin xmax ymax
                box = bfull[i,j]
                class_prediction = classes[i,j]
                score_prediction = scores[i,j]
                f.write('%d %d %d %d %d %f \n' % \
                    (box[0],box[1],box[2],box[3],int(class_prediction),score_prediction))


    # Set up the dataframe from the txt file
    data = pandas.read_csv(file_name, header=-1)
    data.columns = ['init']
    new = data["init"].str.split(" ", expand=True) 
    data["xmin"]= new[0].astype(int)
    data["ymin"]= new[1].astype(int)
    data["xmax"]= new[2].astype(int)
    data["ymax"]= new[3].astype(int)
    data["class_prediction"]= new[4].astype(int)
    data["score_prediction"]= new[5].astype(float)
    data.drop(["init"], axis=1, inplace = True)

    # Confidence 
    data_con = data[data.score_prediction >= score_prediction]

    # Set up the picture and draw the boxes
    orgImg = cv2.imread(input)
    
    # Save the image correctly
    cv2.imwrite(img_name + '_' + model_name + '.png', orgImg)
    name = img_name + '_' + model_name + '_' + 'preds.txt'
    imagename = img_name + '_' + model_name + '.png'
    return name, imagename
