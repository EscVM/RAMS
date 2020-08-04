# Lint as: python
#
# Authors: Vittorio | Francesco
# Location: Turin, Biella, Ivrea
#
# This file is based on the work of Francisco Dorr - PROBA-V-3DWDSR (https://github.com/frandorr/PROBA-V-3DWDSR)

"""Some important functions for preprocessing the original probav-dataset"""
import cv2
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
from scipy.ndimage import shift
from skimage.transform import rescale
from skimage.feature import masked_register_translation



def load_dataset(base_dir, part, band):
    """
    Load the original proba-v dataset already splitted in train, validation and test
    
    Parameters
    ----------
    base_dir: str
        path to the original dataset folder
    part: str
        'train', 'val' or test string
    band: str
        string with the band 'NIR' or 'RED'
    """
    imgsets = sorted(glob(base_dir+"/"+part+"/"+band+"/*"))
    
    X = []; X_masks = []; y = []; y_masks = []
    for imgset in tqdm(imgsets):
        LRs = sorted(glob(imgset+"/LR*.png"))
        QMs = sorted(glob(imgset+"/QM*.png"))
        T = len(LRs)
        
        LR = np.empty((128,128,T),dtype="uint16")
        QM = np.empty((128,128,T),dtype="bool")
        
        for i,img in enumerate(LRs):
            LR[...,i] = cv2.imread(img,cv2.IMREAD_UNCHANGED)
        for i,img in enumerate(QMs):
            QM[...,i] = cv2.imread(img,cv2.IMREAD_UNCHANGED).astype("bool")
        
        X.append(LR)
        X_masks.append(QM)
        
        if part != "test":
            y.append(cv2.imread(imgset+"/HR.png",cv2.IMREAD_UNCHANGED)[...,None])
            y_masks.append(cv2.imread(imgset+"/SM.png",cv2.IMREAD_UNCHANGED).astype("bool")[...,None])
    
    if part != "test":
        return X,X_masks,np.array(y),np.array(y_masks)
    else:
         return X,X_masks



def augment_dataset(X, y, y_masks, n_augment=7):
    """
    Augment the input tensor X of shape (B, H, W, T) and its ground-truths shuffling the temporal channel
    
    Parameters
    ----------
    X: numpy array
        tensor X to augment
    y: numpy array
        tensor y (ground-truths) to augment
    y_mask: numpy array
        tensor y_mask with quality masks of y to augment
    n_augment: int
        augmentation multiplier
    """
    X_aug = np.empty((X.shape[0]*(n_augment),)+X.shape[1:])
    y_aug = np.empty((y.shape[0]*(n_augment),)+y.shape[1:])
    y_masks_aug = np.empty((y_masks.shape[0]*(n_augment),)+y_masks.shape[1:])
        
    for i in tqdm(range(len(X))):
        X_aug[i*n_augment:(i+1)*n_augment],y_aug[i*n_augment:(i+1)*n_augment],y_masks_aug[i*n_augment:(i+1)*n_augment] = \
        augment_imgset(X[i],y[i],y_masks[i],n_augment)
        
    return X_aug,y_aug,y_masks_aug
    
    
    
def augment_imgset(X_imgset, y_imgset, y_mask_imgset, n_augment):
    """
    Augment the input tensor X_imgset of shape (H, W, T) and its ground-truths shuffling the temporal channel
    
    Parameters
    ----------
    X_imgset: numpy array
        tensor X to augment
    y_imgset: numpy array
        tensor y (ground-truths) to augment
    y_mask_imgset: numpy array
        tensor y_mask with quality masks of y to augment
    n_augment: int
        augmentation multiplier
    """
    X_imgset = np.expand_dims(X_imgset,0)      #1,128,128,T
    X_imgset_aug = X_imgset.copy()             #1,128,128,T
    
    T = X_imgset.shape[-1]
    for i in range(n_augment-1):
        permutated_indexes = np.random.permutation(T)
        X_imgset_aug = np.concatenate([X_imgset_aug,X_imgset[...,permutated_indexes]])
    return X_imgset_aug, np.array([y_imgset]*(n_augment)), np.array([y_mask_imgset]*(n_augment))
        


def register_dataset(X, masks):
    """
    Register the input tensor X of shape (B, H, W, T) with respect to the image with the best quality map
    
    Parameters
    ----------
    X: numpy array
        tensor X to register
    masks: numpy array
        tensor with the quality maps of X
    """
    X_reg = []
    masks_reg = []
    
    for i in tqdm(range(len(X))):
        img_reg,m_reg = register_imgset(X[i], masks[i])
        X_reg.append(img_reg)
        masks_reg.append(m_reg)
    
    return X_reg,masks_reg



def register_imgset(imgset, mask):
    """
    Register the input tensor imgset of shape (H, W, T) with respect to the image with the best quality map
    
    Parameters
    ----------
    imgset: numpy array
        imgset to register
    masks: numpy array
        tensor with the quality maps of the imgset
    """
    ref = imgset[...,np.argmax(np.mean(mask,axis=(0,1)))] #best image
    imgset_reg = np.empty(imgset.shape)
    mask_reg = np.empty(mask.shape)
    
    for i in range(imgset.shape[-1]):
        x = imgset[...,i]; m = mask[...,i]
        s = masked_register_translation(ref, x, m)
        x = shift(x, s, mode='reflect')
        m = shift(m, s, mode='constant', cval=0)
        imgset_reg[...,i] = x
        mask_reg[...,i] = m
        
    return imgset,mask_reg


    
def select_T_images(X, masks, T=9, thr=0.85, remove_bad=True):
    """
    Select the best T images of each imgset in X
    
    Parameters
    ----------
    X: numpy array
        tensor X with all scenes
    masks: numpy array
        tensor with the quality maps of all imgset in X
    T: int
        number of temporal steps to select
    thr: float
        percentage for the quality check
    remove_bad: bool
        remove bad timesteps
    """
    X_selected = []
    masks_selected = []
    remove_indexes = []
    
    for i in tqdm(range(len(X))):
        imgset = X[i]; m = masks[i]
        clearance = np.mean(m,axis=(0,1))
        clear_imgset = imgset[...,clearance > thr]
        clear_m = m[...,clearance > thr]
        clearance = clearance[clearance > thr]
        if not len(clearance):
            if remove_bad: #imgset all under threshold, removed
                print("Removing number",i)
                remove_indexes.append(i)
                continue     
            else: # for testing images, take best image
                best_index = np.argmax(np.mean(m,axis=(0,1)))
                clearance = np.mean(m,axis=(0,1))[best_index:best_index+1]
                clear_imgset = imgset[...,best_index:best_index+1]
                clear_m = m[...,best_index:best_index+1]
                
        
        sorted_clearances_indexes = list(np.argsort(clearance)[::-1])    #sort decrescent
        delta = T - len(clearance)

        if delta>0:  # repeat random indexes because we have less than T
            random_indexes = []
            for _ in range(delta):
                random_indexes.append(np.random.choice(sorted_clearances_indexes))
            sorted_clearances_indexes += random_indexes
        
        X_selected.append(clear_imgset[...,sorted_clearances_indexes[:T]])   #take T images        
        masks_selected.append(clear_m[...,sorted_clearances_indexes[:T]])    #take T masks  
            
    return np.array(X_selected), remove_indexes

def sub_images(X,d,s,n): 
    """
    Generate patches util
    """
    l = n**2
    ch = X.shape[-1]
    k = np.empty((l,d,d,ch))
    
    for i in range(n):
        for j in range(n):
            sub = X[i*s:i*s+d,j*s:j*s+d]
            k[n*i+j] = sub
    return k

def gen_sub(array,d,s,verbose=True):
    """
    Generate patches 
    
    Parameters
    ----------
    array: numpy array
        tensor X with all scenes
    d: int
        dimension of the pathches
    s: int
        stride between pathces
    verbose: bool
        print output info
    """
    if len(array.shape) != 4: raise ValueError("Wrong array shape.")
    
    l = len(array)
    d_o = array.shape[1]
    ch = array.shape[-1]
    d = int(d)
    s = int(s)
    
    n = (d_o - d)/s + 1
    if int(n) != n: raise ValueError("d, s and n should be integer values.")
    
    n = int(n)
    
    X_sub = np.empty((l*(n**2),d,d,ch))
    with tqdm(total=l,desc="Creating sub images") if verbose else no_bar() as pbar:
        for i,X in enumerate(array):    
            sub = sub_images(X,d,s,n)
            X_sub[i*n**2:(i+1)*n**2] = sub
            if pbar:
                pbar.update(1)
    if verbose:
        print(X_sub.shape)
    return X_sub

def bicubic(X, scale = 3):
    """
    Rescale with bicubic operation
    
    Parameters
    ----------
    X: numpy array
        tensor X to upscale
    scale: int
        scale dimension
    """
    if len(X.shape) == 3:
        X = np.expand_dims(X,axis=0)
    if len(X.shape) != 4: raise ValueError("Wrong array shape.")
    shape = [X.shape[0],X.shape[1]*scale,X.shape[2]*scale,X.shape[-1]]
    
    X_upscaled = np.empty(shape)
    
    for i,lr in enumerate(X):
        sr_img = rescale(lr,scale=scale,order=3,mode='edge',
                     anti_aliasing=False, multichannel=True, preserve_range=True) #bicubic
        X_upscaled[i] = sr_img

    return X_upscaled
