#!/usr/bin/env python
# coding: utf-8


import os
import glob
import shutil
import fnmatch
import re
import cv2
import csv
import numpy as np 
import pandas as pd
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import ImageFile
import torch
import torch.nn as nn
from model import MyNet
from matplotlib import pyplot as plt
from yattag import Doc
#%matplotlib inline
regex = re.compile(r'(\d+)( \((.+)\))*.jpg')

#classes for defect banknote
classes = {
    'Dot': 'dot',
    'Miss_print': 'miss_print',
    'Over_ink': 'over_ink',
    'Set_off': 'set_off',
    'Under_ink': 'under_ink',
    'Wiping': 'wiping',
}

class_names = ['dot', 'miss_print', 'over_ink', 'set_off', 'under_ink', 'wiping']

# instantiate the CNN
use_cuda = torch.cuda.is_available()
layer_sizes = [512, 256, 128]
model = MyNet(output_size=6, layer_sizes=layer_sizes)
if use_cuda:
    model = model.cuda()

model.load_state_dict(torch.load('model/model_resnet101_512_256_128_back.pt', map_location=torch.device('cpu')))

#input the directory path
dir_path = input('Enter BACK images directory path: ').strip()

#print(dir_path)


# ### Combine images side by side


def merge_images():
    #create output folder
    os.mkdir(dir_path + '/output')
    
    #create classes folder
    for c in class_names:
        os.mkdir(dir_path +'/output/'+c)

    img_id = 0
    files = [f for f in glob.glob(dir_path + '/*.jpg')]
    files = [f for f in files if '_std' not in f]
    files.sort()
    #print (files)
    for fpath in files:
        fname = fpath.split('/')[-1]
        std_fpath = fpath[:-4]+'_std.jpg'
        #Read defect and standard images and combine them
        im1 = cv2.imread(fpath)
        im2 = cv2.imread(std_fpath)
        im3 = cv2.hconcat([im1, im2])

        img_id += 1
        new_fpath = dir_path + '/output/'+str(img_id)+'.jpg'

        cv2.imwrite(new_fpath, im3)


# ### Predict the image class

def predict_image(model, img_path, use_cuda, class_names):
    # load the image and return the predicted breed

    mytransform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    np_img = cv2.imread(img_path)
    #plt.imshow(np_img)

    tensor_img = mytransform(np_img)
    tensor_img = tensor_img.unsqueeze(0) # create tensor with batch dimension
    if use_cuda:
        tensor_img = tensor_img.cuda()
        
    model.eval()
    output = model(tensor_img)
    output = output.cpu().detach().numpy()
    pred_idx = np.argmax(output)
    return class_names[pred_idx]


# ### Predict each file and move to respective folder

def classify_image():
    file_lists = [f for f in glob.glob(dir_path +'/output/*.jpg')]
    for file_path in file_lists:
        file_name = file_path.split('/')[-1]

        cs = str(predict_image(model, file_path, use_cuda, class_names))    
        
        for c in class_names:
  
            if cs == c:
                shutil.move(file_path, dir_path +'/output/'+c+'/'+file_name)



merge_images()
classify_image()


#Count file in folders and put in dictionary
class_output = {}
for c in class_names:
        dir_name = dir_path + '/output/'+c+'/'
        file_count =  glob.glob(dir_name+'*.*')
        #print(c,len(file_count))
        class_output[c] = len(file_count)
#print(class_output)        


#write dictionary file to CSV
csv_file = 'output.csv'
csv_columns = ['Type', 'Total Number']

with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for key, value in class_output.items():
                writer.writerow({'Type': key, 'Total Number': value})



#Read CSV and plot bar graph
df = pd.read_csv('output.csv', sep=',')
print(df)

df.set_index('Type').plot.bar(legend=None)

for i, val in enumerate(df['Total Number'].values):
    plt.text(i, val+0.1, df['Total Number'][i])

plt.title('Unfit Back Banknotes Classification by Number')
plt.xlabel('Classifications')
plt.xticks(rotation=0)
plt.ylabel('Number of Notes')
plt.savefig(dir_path+'/output/output.png')


print('Classification done!, the images are sorted to the respective folder')


# ### Generate the HTML ouput

doc, tag, text = Doc().tagtext()


with tag('div', id='photo_container'):
    doc.stag('img', src='output.png', klass='photo')
    doc.stag('br')
with tag('h2'):
    text('Classification Ouput')
with tag('a', href='./dot/'):
    text('Output folder for DOT')
    doc.stag('br')
with tag('a', href='./miss_print/'):
    text('Output folder for Miss print')
    doc.stag('br')
with tag('a', href='./over_ink/'):
    text('Output folder for Over ink')
    doc.stag('br')
with tag('a', href='./set_off/'):
    text('Output folder for Set off')
    doc.stag('br')
with tag('a', href='./under_ink/'):
    text('Output folder for Under ink')
    doc.stag('br')
with tag('a', href='./wiping/'):
    text('Output folder for Wiping')
    doc.stag('br')
    

f = open(dir_path+'/output/output.html','w')

message = '''<html>
<head></head>
<body>'''+ doc.getvalue() + '''</body>
</html>'''

f.write(message)
f.close()


print('The report is generated to output.html')



