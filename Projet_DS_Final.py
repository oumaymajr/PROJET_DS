#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import ast
from tqdm import tqdm
from random import randint
from PIL import ImageEnhance, ImageOps

from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image
import pandas as pd
import cv2 as cv
import os
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


origine_image = cv.imread('C:/Users/Admin/Desktop/data_org/axis.png')
print(origine_image.shape)


# In[ ]:


# data preprocessing : (resize,grayscale,remove noise)
def prepImg(img,index):
    aa = Image.open(img)
    aa = aa.resize((512, 256)) # shape of image
    gaa = ImageOps.grayscale(aa)
    pixa = np.array(gaa.getdata()).reshape(aa.size[0], aa.size[1], 1)
    flatPix = pixa.flatten()

    for i in np.arange(0,aa.size[0]):
        for j in np.arange(0,aa.size[1]):
            if pixa[i][j]>170:
                pixa[i][j]=255     
    pixa = pixa.flatten()
    pixa = np.reshape(pixa, (256, 512))
    dt = Image.fromarray((pixa).astype(np.uint8))
    dt1 = ImageEnhance.Sharpness(dt)
    index = str(index)
    for i in np.arange(0,6-len(index)):
        index = "0"+index
    dt1.enhance(2).save("C:/Users/Admin/Desktop/data_preparation/"+index+".jpg")


# In[ ]:


#apply the treatment on the file
import os
path="C:/Users/Admin/Desktop/data_org/"
files = os.listdir(path) #a list containing the names of the entries in the directory given by path

index = 0
for f in files:
#for i in np.arange(0,4):
    if (f[-3:] != "csv"):
        f = path+f
        prepImg(f,index)
        index=index+1


# In[ ]:


get_ipython().run_line_magic('cd', 'C:/Users/Admin/Desktop/yolov5')


# In[ ]:


get_ipython().system('python detect.py --source C:/Users/Admin/Desktop/data_preparation/ --weights C:/Users/Admin/Desktop/yolov5/runs/train/essaie12/weights/best.pt --img 512 --save-txt --save-conf --save-crop --line-thickness=1 --augment --dnn ')


# <font color=Red><h1>AutoCorrection</h1> </font>

# # Analyse Lexicale

# In[ ]:


#AIL : AmountInLetters
AILList = amountInLetters.split(" ")
for i in np.arange(0,len(AILList)):
    AILList[i] = AILList[i].upper()


# In[ ]:


keyWordsList = [['ZERO', 'ZERO', 0, True], ['ONE', 'UN', 1, True], ['TWO', 'DEUX', 2, True], ['THREE', 'TROIS', 3, True], ['FOUR', 'QUATRE', 4, True], ['FIVE', 'CINQ', 5, True], ['SIX', 'SIX', 6, True], ['SEVEN', 'SEPT', 7, True], ['EIGHT', 'HUIT', 8, True], ['NINE', 'NEUF', 9, True], ['TEN', 'DIX', 10, True], ['ELEVEN', 'ONZE', 11, True], ['TWELVE', 'DOUZE', 12, True], ['THIRTEEN', 'TREIZE', 13, True], ['FOURTEEN', 'QUATORZE', 14, True], ['FIFTEEN', 'QUINZE', 15, True], ['SIXTEEN', 'SEIZE', 16, True], ['SEVENTEEN', 'DIX-SEPT', 17, True], ['EIGHTEEN', 'DIX-HUIT', 18, True], ['NINETEEN', 'DIX-NEUF', 19, True], ['TWENTY', 'VINGT', 20, True], ['THIRTY', 'TRENTE', 30, True], ['FORTY', 'QUARANTE', 40, True], ['FIFTY', 'CINQUANTE', 50, True], ['SIXTY', 'SOIXANTE', 60, True], ['SEVENTY', 'SOIXANTE-DIX', 70, True], ['EIGHTY', 'QUATRE-VINGTS', 80, True], ['NINETY', 'QUATRE-VINGT-DIX', 90, True], ['AND', 'ET', 0, False], ['HUNDRED', 'CENT', 100, False], ['THOUSAND', 'MILLE', 1000, False], ['MILLION', 'MILLION', 1000000, False],['DOLLARS', 'DOLLARS', 0, True],['DINARS', 'DINARS', 0, True],['EUROS', 'EUROS', 0, True]]


# In[ ]:


def checkIfInt(variable):
    try:
        aux = int(variable)
        return aux
    except:
        return False


# In[ ]:


def getIndex(array,element):
    for i in np.arange(0,len(array)):
        if (array[i][lang] == element):
            #break;
            return i
    return -1


# In[ ]:


def levenshteinDistance(a, b):

    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),      # insert character
            min_dist(s1 + 1, s2),      # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)


# In[ ]:


def spellCheck(Element):
    closestWord = [-1,1000] #index = -1, distance = 1000
    for j in np.arange(0,len(keyWordsList)):
        dist = levenshteinDistance(Element,keyWordsList[j][lang])
        if (dist <= closestWord[1]):
            closestWord = [j,dist]
    return keyWordsList[closestWord[0]][lang]


# In[ ]:


#LTN : letterToNumber
def LTNConversion():
    convertedAmount = aux = indexAux = 0
    errorFlag = False
    for i in np.arange(0,len(AILList)):
        indexAux = getIndex(keyWordsList,AILList[i])
        if (indexAux != -1):
            if (keyWordsList[indexAux][len(keyWordsList[0])-1] == True ):
                aux = aux + keyWordsList[indexAux][len(keyWordsList[0])-2]
            else:
                convertedAmount = convertedAmount + aux * keyWordsList[indexAux][len(keyWordsList[0])-2]
                aux = 0
        else:
            print("Error : Unknown element to keywordsList found")
            errorFlag = True
    convertedAmount = convertedAmount + aux
    if (errorFlag != True):
        return(convertedAmount)


# In[ ]:


changes = [] #an array that contains all the spellchecked words [[oldWord, spellCheckedWord], ...]
for i in np.arange(0,len(AILList)):
    if(getIndex(keyWordsList,AILList[i]) == -1): #if element doesn't exist in keywords spellcheck it
        changes.append([AILList[i],spellCheck(AILList[i])])
        AILList[i] = spellCheck(AILList[i])
convertedAmount = LTNConversion() #amount equal to the given amount in letters


# In[ ]:


print(convertedAmount,AILList,changes)


# # Analyse Sémantique

# In[ ]:


convertedAmountLength = len(str(convertedAmount))
amountInDigitsLength = len(amountInDigits)
errors = [0]
intAmountInDigits = checkIfInt(amountInDigits)
a = b = ""
if (intAmountInDigits): #check if amount in digits in an integer
    if (amountInDigitsLength != convertedAmountLength):
        errors[0] = errors[0] + abs(convertedAmountLength-amountInDigitsLength)
        errors.append(["different size","same size"])
        print("Size Exceeded")
        amountInDigits = amountInDigits[:len(str(min(intAmountInDigits,convertedAmount)))] 
    for i in np.arange(0,len(amountInDigits)):
        if (amountInDigits[i] != str(convertedAmount)[i]):
            errors.append([amountInDigits[i],str(convertedAmount)[i]])
            errors[0] = errors[0] + 1
    if (errors[0]>2):
        print("More than 2 mistakes were detected, cheque is revoked")
    else:
        for j in np.arange(1,len(errors)):
            a = a + errors[j][0]
            b = b + errors[j][1]
        print("cheque contained " + str(errors[0]) + " semantic mistakes : expected " + b + " found " + a)
        print("and " + str(len(changes)) + " syntaxic mistakes :")
        for k in np.arange(0,len(changes)):
            print(changes[k][0] + " => " + changes[k][1])
else:
    print("Error : Amount in digits contains letters")


# In[ ]:


string =""
for i in np.arange(0,len(AILList)):
    string = string + AILList[i] + " " 
print(string)
print(convertedAmount)

