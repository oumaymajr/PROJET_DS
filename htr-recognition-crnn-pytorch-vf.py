#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('git clone https://github.com/githubharald/CTCDecoder.git')


# In[9]:


get_ipython().run_line_magic('cd', 'C:/Users/Admin/Downloads/CTCDecoder')


# In[6]:


get_ipython().system('pip install albumentations')


# In[7]:


import albumentations
import torch
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt
from tqdm import tqdm
import string
from skimage import io
import os
import glob

from ctc_decoder import best_path, beam_search
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence
import torchvision.models as models
from torch.utils.model_zoo import load_url
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
import cv2
import torch
from  torch import nn
from torch.nn import functional as F
from pprint import  pprint


# In[4]:


aymen_csv = pd.read_csv('../input/handwrittendata/resWords.csv')
aymen_csv["image_path"] = aymen_csv["IMAGE_ID"].apply(lambda x: f"../input/handwrittendata/Words/"+x)


# In[5]:


aymen_csv["image_path"]


# In[6]:


upper = list(string.ascii_uppercase)
numbers = [i for i in range(10)]
aymen_csv["LABEL TYPE"] = aymen_csv["LABEL"].apply(lambda x: "phrase" if x[0] in upper else "numbers")


# In[7]:


#train_csv = aymen_csv.loc[aymen_csv['LABEL TYPE'] == "phrase"]


# In[8]:


train_csv = aymen_csv


# In[9]:


train_csv


# In[10]:


train_csv = train_csv[['IMAGE_ID', 'LABEL','image_path']]


# In[11]:


train_csv


# In[12]:


#train_csv = aymen_csv[['IMAGE_ID', 'LABEL']]
#_, train_csv = model_selection.train_test_split(train_csv,
#                                test_size=0.99,
#                                shuffle=True,
#                                random_state=42
#                                #stratify=aymen_csv["LABEL TYPE"]
#                               )


# In[13]:


train_csv = train_csv.reset_index(drop=True)


# In[14]:


train_csv


# In[15]:


#train_csv.to_csv('train.csv', index=False)


# In[16]:


#train_df = pd.read_csv('./train.csv')
#train_df['LABEL']


# In[17]:


#line_df = train_df
#root_dir='../input/fulldataletterchiffrehandwirting/Words/Words'
#for idx in tqdm(range(0, len(line_df))):
#    # image name 
#    img_name = line_df.iloc[idx, 0]
#    # Define image path
#    img_filepath = os.path.join(f"{root_dir}/{img_name}")
#    #print(img_filepath)
#    # Read target image gray scale
#    image = cv2.imread(img_filepath)
#    #print(image_gray.shape)
#    # write the image in dataset directory
#    cv2.imwrite(f'./dataset/{img_name}', image)


# In[10]:


img = io.imread('C:/Users/Admin/Desktop/DATA_DIR')
plt.imshow(img)
img.shape


# In[11]:


class config:
    DATA_DIR = 'C:/Users/Admin/Desktop/DATA_DIR'
    BATCH_SIZE = 64
    IMAGE_WIDTH = 500
    IMAGE_HEIGHT = 100
    NUM_WORKERS = 2
    EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )


# In[20]:


# define as global 
image_paths = train_csv['image_path']
image_files = train_csv['LABEL']
# "/../../sdfez.png"
targets_orig = [x for x in image_files]
# abcde -> [a, b, c, d, e]
targets = [[c for c in x] for x in targets_orig]
#targets = targets + [['$','$','$','$','$','$','$']]
#for target in targets:
#    while len(target) < 10:
#        target.append(" ")

targets_flat = [c for clist in targets for c in clist]

lbl_encoder = preprocessing.LabelEncoder()
lbl_encoder.fit(targets_flat)

targets_enc  = [lbl_encoder.transform(x) for x in targets]
targets_enc  = np.array(targets_enc ) 


# In[21]:


lbl_encoder.classes_


# In[22]:


#chars_list = lbl_encoder.classes_.tolist()
#print(len(chars_list))
#chars_list.append('°')
#
#chars = ''.join(chars_list)
#len(chars)


# In[23]:


#targets_enc.shape


# In[24]:


train_csv["LABEL ENCODED"] = targets_enc


# In[25]:


train_csv


# In[26]:


class dataset:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    class ClassificationDataset:
        def __init__(self, dataframe, resize=None):
            self.dataframe = dataframe
            self.resize = resize
            self.aug = albumentations.Compose([albumentations.Normalize(always_apply=True)])

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, item):
            image = Image.open(self.dataframe['image_path'][item]).convert("RGB")
            targets = self.dataframe['LABEL ENCODED'][item]
            if self.resize is not None:
                image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

            image = np.array(image)
            augmented = self.aug(image=image)
            image = augmented["image"]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            return {
                "images": torch.tensor(image, dtype=torch.float),
                "targets": torch.tensor(targets, dtype=torch.long)
            }


# In[27]:


class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear1 = nn.Linear(1600, 64)
        self.drop_1 = nn.Dropout(0.2)

        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images,lens, targets=None):
        bs, c, h, w = images.size()
        #print(bs, c, h, w)
        x = F.relu(self.conv_1(images))
        #print(x.size())
        x = self.pool_1(x)
        #print(x.size())
        x = F.relu(self.conv_2(x))
        #print(x.size())
        x = self.pool_2(x) # 1, 64, 18, 75
        #print(x.size())
        x = x.permute(0, 3, 1, 2) # 1, 75 , 64, 18
        #print(x.size())
        x = x.view(bs, x.size(1), -1)
        #print(x.size())
        x = self.linear1(x)
        x = self.drop_1(x)
        #print(x.size()) # torch.Size([1, 75, 64]) -> we have 75 time steps and for each time step we have 64 values
        x, _ = self.gru(x)
        #print(x.size())
        x = self.output(x)
        #print(x.size())
        x = x.permute(1, 0, 2) # bs, time steps, values -> CTC LOSS expects it to be
        #print(f"x permute :{x.size()}")
        #if targets is not None:
        #    log_probs = F.log_softmax(x, 2)
        #    print(f"log_probs :{log_probs.size()}")
        #    input_lengths = torch.full(
        #        size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
        #    )
        #    print(f"input_lengths : {input_lengths}")
        #    target_lengths = torch.full(
        #        size=(bs,), fill_value=targets.size(1), dtype=torch.int32
        #    )
        #    print(f"target_lengths : {target_lengths}")
        #    loss = nn.CTCLoss(blank=0)(
        #        log_probs, targets, input_lengths, target_lengths
        #    )
        #    return x, loss
        #
        
        # x = bs, time steps, values
        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            #print(input_lengths)
            #target_lengths = torch.full(
            #    size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            #)
            #print(target_lengths)
            target_lengths = torch.tensor(
                lens, dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=len(lbl_encoder.classes_))(
                log_probs, targets, input_lengths, lens
            )
            return x, loss

        return x, None


# In[28]:


#test_img = torch.rand([2, 3, 100, 500])
#test_target = torch.rand([2,40])
#lens = torch.rand([36,41])
#model_test = CaptchaModel(len(lbl_encoder.classes_))


# In[29]:


#model_test(test_img,test_target)


# In[30]:


#test_img = torch.rand([8, 3, 100, 500])
#model_test = CaptchaModel(len(lbl_encoder.classes_))


# In[31]:


#sample_test = train_dataset[0]
#sample_test['images'] = sample_test['images'].unsqueeze(0)
#print(sample_test['images'].shape)


# In[32]:


#image_paths = glob.glob(os.path.join(config.DATA_DIR, "*.jpg"))
#len(image_paths)


# In[33]:


#image_paths = glob.glob(os.path.join(config.DATA_DIR, "*.jpg"))
#image_files = train_df['LABEL']
## "/../../sdfez.png"
#targets_orig = [x for x in image_files]
## abcde -> [a, b, c, d, e]
#targets = [[c for c in x] for x in targets_orig]
#
##for target in targets:
##    while len(target) < 10:
##        target.append(" ")
#
#targets_flat = [c for clist in targets for c in clist]
#
#lbl_encoder = preprocessing.LabelEncoder()
#lbl_encoder.fit(targets_flat)
#
#targets_enc  = [lbl_encoder.transform(x) for x in targets]
#targets_enc  = np.array(targets_enc ) + 1


# In[34]:


#targets_test = [[' '],['a']]
#targets_enc_test  = [lbl_encoder.transform(x) for x in targets_test]
#targets_enc_test


# In[35]:


p = lbl_encoder.inverse_transform([19, 36, 22, 29, 33, 38,  0, 12, 26, 24, 25, 33,  0, 19, 25, 30, 34,
              32, 20, 29, 21,  0, 13, 26, 35, 22,  0, 14, 34, 29, 21, 31, 22, 21,
               0, 20, 29, 21,  0, 13, 30, 31, 33, 38,  0, 19, 36, 30])
p


# In[36]:


#lbl_encoder.transform([' '])


# In[37]:


#image_paths = glob.glob(os.path.join(config.DATA_DIR, "*.jpg"))
#image_files = train_df['LABEL']
## "/../../sdfez.png"
#targets_orig = [x for x in image_files]
## abcde -> [a, b, c, d, e]
#targets = [[c for c in x] for x in targets_orig]
#
##for target in targets:
##    while len(target) < 10:
##        target.append(" ")
#
#targets_flat = [c for clist in targets for c in clist]
#
#lbl_encoder = preprocessing.LabelEncoder()
#lbl_encoder.fit(targets_flat)
#
#targets_enc  = [lbl_encoder.transform(x) for x in targets]
#targets_enc  = np.array(targets_enc ) + 1
#
#(
#    train_imgs,
#    test_imgs,
#    train_targets,
#    test_targets,
#    _,
#    test_targets_orig,
#) = model_selection.train_test_split(
#    image_paths, targets_enc, targets_orig, test_size=0.1, random_state=42
#)


# In[38]:


#train_targets


# In[39]:


def collate(batch):
    images, words = [b.get('images') for b in batch], [b.get('targets') for b in batch]
    images = torch.stack(images, 0)
    # Calculate target lengths for the current batch
    lens = [len(item['targets']) for item in batch]
    
    list_of_chars_digitized = [word for word in words]
    lengths = len(list_of_chars_digitized)
    # According to https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
    # Tensor of size sum(target_lengths) the targets are assumed to be un-padded and concatenated within 1 dimension.
    #targets = torch.empty(sum(lengths)).fill_(len(classes)).long()
    lengths = torch.tensor(lengths, dtype=torch.long)
    lens = torch.tensor(lens, dtype=torch.long)
    #print(words)
    # Now we need to fill targets according to calculated lengths
    for i in range(lengths.item()):
        targets = torch.cat(words)
    
    
    #print(f" lens : {lens}")
    #for j, word in enumerate(words):
    #    start = sum(lengths[:j])
    #    end = lengths[j]
    #    targets[start:start + end] = torch.tensor([dataset.char_dict.get(letter) for letter in word]).long()
    
    
    return  images,targets, lens


# In[40]:


#lbl_encoder.transform([' '])[0]


# In[41]:


#def my_collate(batch):
#    # batch contains a list of tuples of structure (sequence, target)
#    images = [item['images'] for item in batch]
#    images = torch.stack(images, 0)
#    #images = pad_sequence(images)
#    targets = [item['targets'] for item in batch]
#    lens = [len(item['targets']) for item in batch]
#    targets = pad_sequence(targets, batch_first=True, padding_value=lbl_encoder.transform([' '])[0])
#    
#    return {
#        "images": images,
#        "targets": targets,
#        "lens" : lens
#           }


# In[42]:


#(
#        train,
#        test,
#        _,
#        test_targets_orig,
#    ) = model_selection.train_test_split(
#        train_csv, targets_orig, test_size=0.1, random_state=42
#    )
#
#train_imgs = train_imgs.reset_index(drop=True)
#test_imgs = test_imgs.reset_index(drop=True)
#
#train_dataset = dataset.ClassificationDataset(
#        dataframe=train,
#        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
#    )
#train_loader = torch.utils.data.DataLoader(
#        train_dataset,
#        batch_size=config.BATCH_SIZE,
#        num_workers=config.NUM_WORKERS,
#        shuffle=True,
#        collate_fn=my_collate
#    )
#test_dataset = dataset.ClassificationDataset(
#        dataframe=test,
#        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
#        
#    )
#test_loader = torch.utils.data.DataLoader(
#        test_dataset,
#        batch_size=config.BATCH_SIZE,
#        num_workers=config.NUM_WORKERS,
#        shuffle=False,
#        collate_fn=my_collate
#    )


# In[43]:


#plt.imshow(train_dataset[0]['images'].permute(1,2,0))
#train_dataset[0]['targets']


# In[44]:


targets_enc


# In[45]:


p = lbl_encoder.inverse_transform([31, 60, 42, 51, 57, 62,  0, 30, 46, 61,  0, 14, 55, 52, 55, 42,  0,
              17, 46, 43, 57, 62,  0, 30, 42, 59, 42, 51,  0, 23, 38, 48, 45])
p


# In[46]:


#train_dataset = dataset.ClassificationDataset(
#            dataframe=train_csv,
#            resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
#        )
#train_loader = torch.utils.data.DataLoader(
#            train_dataset,
#            batch_size=config.BATCH_SIZE,
#            num_workers=config.NUM_WORKERS,
#            shuffle=True,
#            collate_fn=collate
#        )


# In[47]:


#test_batch = next(iter(train_loader))
#test_batch


# In[48]:


#target_lengths = torch.full(
#size=(8,), fill_value=test_batch[2], dtype=torch.long)


# In[49]:


class engine:
    def train_fn(model, data_loader, optimizer):
        model.train()
        fin_loss = 0
        tk0 = tqdm(data_loader, total=len(data_loader))
        for xb, yb, lens in tk0:
        #for data in tk0:
            #for key, value in data.items():
                #data[key] = value.to(config.DEVICE)
            
            xb = xb.to(config.DEVICE)
            yb = yb.to(config.DEVICE)
            lens = lens.to(config.DEVICE)
            
            optimizer.zero_grad()
            _, loss = model(xb, lens, yb)
            loss.backward()
            optimizer.step()
            fin_loss += loss.item()
        return fin_loss / len(data_loader)


    def eval_fn(model, data_loader):
        model.eval()
        fin_loss = 0
        fin_preds = []
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for xb, yb, lens in tk0:
            #for data in tk0:
                #for key, value in data.items():
                    #data[key] = value.to(config.DEVICE)
                
                xb = xb.to(config.DEVICE)
                yb = yb.to(config.DEVICE)
                lens = lens.to(config.DEVICE)
                
                batch_preds, loss = model(xb, lens, yb)
                fin_loss += loss.item()
                fin_preds.append(batch_preds.detach().cpu())
            return fin_preds, fin_loss / len(data_loader)


# In[50]:


def decode_predictions(preds, encoder):
    #print(preds.shape) # [125, 8, 30] timesteps ,bs , classes
    preds = preds.permute(1, 0, 2) 
    preds = torch.softmax(preds, 2)
    #print(preds.shape) # [8, 125, 30] bs, timesteps, classes
    preds = torch.argmax(preds, 2)
    #print(preds.shape) # [8, 125]
    preds = preds.numpy()
    #print(preds.shape) # (8, 125)
    cap_preds = []
    
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            if k == len(lbl_encoder.classes_):
                temp.append("°")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp)
        cap_preds.append(tp)
    return cap_preds


# In[51]:


#test = [' ', 'C', 'E', 'F', 'H', 'L', 'N', 'O', 'S', 'T', 'a', 'd', 'e',
#       'f', 'g', 'h', 'i', 'k', 'l', 'n', 'o', 'r', 's', 't', 'u', 'v',
#       'w', 'x','y']


# In[52]:


def decode_predictions_2(preds, encoder):
    #print(preds.shape) # [125, 8, 30] timesteps,bs , classes
    #preds = preds.permute(1, 0, 2) 
    #print(preds)
    preds = torch.softmax(preds, 2)
    #print(preds.shape) # [8, 125, 30] bs, timesteps, classes
    
    #preds = torch.argmax(preds, 2)
    #print(preds.shape) # [8, 125]
    preds = preds.detach().cpu().numpy()
    #print(preds.shape) # (8, 125)

    #print(preds.shape)
    for i in range(preds.shape[1]):
        #print(preds[:,i,:].shape)
        #print(preds[:,i,:])
        aux =  preds[:,i,:] 
        #print(aux.shape) # (125, 30)
        #print(f'Best path: "{best_path_1(aux, test)}"')
        print(f'Beam search: "{beam_search(aux,lbl_encoder.classes_)}"')
        
    
    return None


# In[97]:


(
    train,
    test,
    _,
    test_targets_orig,
) = model_selection.train_test_split(
    train_csv, targets_orig, test_size=0.1, random_state=42
)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

train_dataset = dataset.ClassificationDataset(
        dataframe=train,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
test_dataset = dataset.ClassificationDataset(
        dataframe=test,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        
    )


# In[162]:


train_dataset[0]


# In[163]:


p = lbl_encoder.inverse_transform([25, 46, 51, 42, 57, 42, 42, 51,  0, 31, 45, 52, 58, 56, 38, 51, 41,  0,
         31, 60, 52,  0, 19, 58, 51, 41, 55, 42, 41,  0, 38, 51, 41,  0, 31, 45,
         46, 55, 57, 62,  0, 31, 60, 52])
p


# In[54]:


train_dataset[0]['images'].shape


# In[55]:


test_image = train_dataset[0]['images'].permute(1,2,0)
test_image = np.array(test_image)
plt.imshow(test_image)


# In[56]:


train_dataset[0]['targets']


# In[57]:


p = lbl_encoder.inverse_transform([25, 46, 51, 42, 57, 42, 42, 51,  0, 31, 45, 52, 58, 56, 38, 51, 41,  0,
        31, 60, 52,  0, 19, 58, 51, 41, 55, 42, 41,  0, 38, 51, 41,  0, 31, 45,
        46, 55, 57, 62,  0, 31, 60, 52])
p


# In[58]:


def run_training():
    
    
    
    (
        train,
        test,
        _,
        test_targets_orig,
    ) = model_selection.train_test_split(
        train_csv, targets_orig, test_size=0.1, random_state=42
    )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    train_dataset = dataset.ClassificationDataset(
            dataframe=train,
            resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        )
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=True,
            collate_fn=collate
        )
    test_dataset = dataset.ClassificationDataset(
        dataframe=test,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        collate_fn=collate
    )

    model = CaptchaModel(num_chars=len(lbl_encoder.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    best_score = 0.0 
    best_loss = np.inf

    for epoch in range(config.EPOCHS):
        
        #if epoch == 298:
        #    FILE = "my_checkpoint.pth.tar"
        #    torch.save(model.state_dict(), FILE)
        #    print("=> saving checkpoint")
        #    #checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
        #    #save_checkpoint(checkpoint)
                
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, test_loader)
        valid_cap_preds = []
        for vp in valid_preds:
            decode_predictions_2(vp ,lbl_encoder )
            break
            
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_encoder)
            valid_cap_preds.extend(current_preds)

        pprint(list(zip(test_targets_orig, valid_cap_preds))[0:6])
        print(f"EPOCH: {epoch}   ,    train_loss={train_loss},    valid_loss={valid_loss}")
        

            
        if valid_loss < best_loss:
            best_loss = valid_loss
            print("=> saving checkpoint")
            torch.save(model.state_dict(),'Words_best_score.pt')


# In[59]:


#run_training()


# In[60]:


#valid_preds, valid_loss = engine.eval_fn(test_model, test_loader)
#valid_cap_preds = []
#for vp in valid_preds:
#    #print(vp.shape) # [125, 8, 30]
#    #for single_preds in vp:
#        #single_preds = single_preds.T
#        #print(single_preds.detach().cpu().numpy().shape)
#        #print(f'Best path: "{best_path(single_preds.detach().cpu().numpy(), chars)}"')
#    decode_predictions_2(vp ,lbl_encoder )
#    #current_preds = decode_predictions(vp, lbl_encoder)
#    #valid_cap_preds.extend(current_preds)
#    
##pprint(list(zip(test_targets_orig, valid_cap_preds))[6:11])
##print(f"valid_loss={valid_loss}")


# In[61]:


test_model = CaptchaModel(num_chars=len(lbl_encoder.classes_))
test_model.to(config.DEVICE)


# In[62]:


len(lbl_encoder.classes_)


# In[63]:


PATH = '../input/modelvf/Words_best_score VF.pt'
test_model.load_state_dict(torch.load(PATH))


# In[92]:


test_image.shape


# ## inference 

# In[230]:


class config:
    DATA_DIR = '../input/handwrittendata/Words'
    BATCH_SIZE = 64
    IMAGE_WIDTH = 500
    IMAGE_HEIGHT = 100
    NUM_WORKERS = 2
    EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )


# In[241]:


img = Image.open('../input/handwrittendata/Words/10006.jpg').convert("RGB")


# In[242]:


img


# In[243]:


img = img.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT), resample=Image.BILINEAR)


# In[244]:


img = np.array(img)


# In[245]:


aug = albumentations.Compose([albumentations.Normalize(always_apply=True)])
augmented = aug(image=img)
img = augmented["image"]


# In[246]:


img = torch.tensor(img, dtype=torch.float)


# In[247]:


img = img.permute(2,0,1)


# In[248]:


img = img.unsqueeze(0)
img = img.to(config.DEVICE)


# In[249]:


test_model.eval()
with torch.no_grad():
    result_rnn,_ = test_model(img,_)


# In[250]:


def decode_predictions_3(preds, encoder):
    #print(preds.shape) # [125, 8, 30] timesteps,bs , classes
    #preds = preds.permute(1, 0, 2) 
    #print(preds)
    preds = torch.softmax(preds, 2)
    #print(preds.shape) # [8, 125, 30] bs, timesteps, classes
    
    #preds = torch.argmax(preds, 2)
    #print(preds.shape) # [8, 125]
    preds = preds.detach().cpu().numpy()
    #print(preds.shape) # (8, 125)

    #print(preds.shape)
    for i in range(preds.shape[1]):
        #print(preds[:,i,:].shape)
        #print(preds[:,i,:])
        aux =  preds[:,i,:] 
        #print(aux.shape) # (125, 30)
        #print(f'Best path: "{best_path_1(aux, test)}"')
        print(f'Beam search: "{beam_search(aux,lbl_encoder.classes_)}"')
        
    
    return None


# In[251]:


decode_predictions_3(result_rnn,lbl_encoder)

