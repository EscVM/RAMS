# Lint as: python
#
# Authors: Vittorio | Francesco
# Location: Turin, Biella, Ivrea
"""Predictionfunctions for testing RAMS"""
import tensorflow as tf
import numpy as np


def ensemble(X, geometric=True, shuffle = False, n=10):
    """RAMS+ prediction util"""
    if geometric:
        return geometric_ensemble(X,shuffle)
    else:
        random_ensemble(X,n,shuffle)

        
def random_ensemble(X,n=10,shuffle=True):
    """RAMS+ prediction util"""
    r = np.zeros((n,2))
    X_ensemble = []   
    for i in range(n):
        X_aug,r[i,0] = flip(X)
        X_aug,r[i,1] = rotate(X_aug)
        if shuffle:
            X_aug = shuffle_last_axis(X_aug)
        X_ensemble.append(X_aug)
    return tf.convert_to_tensor(X_ensemble),r


def geometric_ensemble(X,shuffle=False):
    """RAMS+ prediction util"""
    r = np.array(np.meshgrid([0, 1],[0,1,2,3])).T.reshape(-1,2) # generates all combinations (8) for flip/rotate parameter
    X_ensemble = []   
    for i in range(8):
        X_aug = flip(X,r[i,0])[0]
        X_aug = rotate(X_aug,r[i,1])[0]
        if shuffle:
            X_aug = shuffle_last_axis(X_aug)
        X_ensemble.append(X_aug)
    return tf.convert_to_tensor(X_ensemble),r


def unensemble(X,r):
    """RAMS+ prediction util"""
    X_unensemble = []   
    for i in range(len(X)):
        X_aug = rotate(X[i],4-r[i,1])[0] # to reverse rotation: k2=4-k1
        X_aug = flip(X_aug,r[i,0])[0]
        X_unensemble.append(X_aug.numpy())
    return np.mean(X_unensemble,axis=0,keepdims=True)

       
def flip(X,rn=None):
    """flip a tensor"""
    if rn is None:
        rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn <= 0.5, lambda: X, lambda: tf.image.flip_left_right(X)), np.rint(rn)


def rotate(X,rn=None):
    """rotate a tensor"""
    if rn is None:
        rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(X, rn), rn

                   
def shuffle_last_axis(X):
    """shuffle last tensor axis"""
    X = tf.transpose(X)
    X = tf.random.shuffle(X)
    X = tf.transpose(X)       
    return X


def predict_tensor(model, x):  
    """RAMS prediction util"""
    lr_batch = tf.cast(x, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 2**16)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.float32)
    return sr_batch


def predict_tensor_permute(model, x, n_ens=10):  
    """RAMS+ prediction util"""
    lr_batch = tf.cast(x, tf.float32)
    sr_batch = []
    for _ in range(n_ens):
        lr = shuffle_last_axis(lr_batch)
        sr_batch.append(model(lr[None])[0])
    sr_batch = tf.convert_to_tensor(sr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 2**16)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.float32)
    return np.mean(sr_batch.numpy(),axis=0,keepdims=True)

def savePredictions(x, band, submission_dir):
    """RAMS save util"""
    if band == 'NIR':
        i = 1306
    elif band=='RED':
        i = 1160
        
    for index in tqdm(range(len(x))):
        io.imsave(os.path.join(submission_dir, f'imgset{i}.png'), x[index][0,:,:,0].numpy().astype(np.uint16),
                  check_contrast=False)
        i+=1

def savePredictionsPermut(x, band, submission_dir):
    """RAMS+ save util"""
    if band == 'NIR':
        i = 1306
    elif band=='RED':
        i = 1160
        
    for index in tqdm(range(len(x))):
        io.imsave(os.path.join(submission_dir, f'imgset{i}.png'), x[index][0,:,:,0].astype(np.uint16),
                  check_contrast=False)
        i+=1