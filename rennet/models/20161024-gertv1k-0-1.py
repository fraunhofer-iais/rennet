#!/usr/bin/python
# coding: utf-8

# # Initializations

# In[2]:

from __future__ import division, print_function

import os
import sys
rennetroot = os.environ['RENNET_ROOT']
sys.path.append(rennetroot)


import numpy as np
import glob
import h5py
from sklearn.metrics import confusion_matrix
from scipy.signal import medfilt
from collections import Counter
import time

import keras.layers as kl
import keras.optimizers as ko
from keras.models import Sequential
from keras.utils import np_utils
import keras.callbacks as kc

np.random.seed(3429342)


# # Read data from pickles

# ## gertv1000

# In[5]:

working_dir = os.path.join(rennetroot, "data", "working")

gertv_proj = "gertv1000-utt"
gertv_data = "AudioMining"

gertv_trn_fps = glob.glob(os.path.join(working_dir, gertv_proj, gertv_data, "train", "pickles", "20161019*.hdf5"))
print(len(gertv_trn_fps))


# In[11]:

ger_X_trn = []
ger_y_trn = []
for f in gertv_trn_fps:
    trn_f = h5py.File(f, 'r')
    ger_X_trn.append(trn_f["data"][()])
    ger_y_trn.append(trn_f["labels"][()])
ger_X_trn = np.concatenate(ger_X_trn)
ger_y_trn = np.concatenate(ger_y_trn)

print("Read data")
print(ger_X_trn.shape, ger_y_trn.shape)


# In[12]:

ger_train_till = int(0.95 * ger_y_trn.shape[0])

print("Validation split")
print(Counter(ger_y_trn[ger_train_till:]).most_common())


# In[13]:

ger_X_val = ger_X_trn[ger_train_till:, :]
ger_y_val = ger_y_trn[ger_train_till:]

ger_X_trn = ger_X_trn[:ger_train_till, :]
ger_y_trn = ger_y_trn[:ger_train_till]

print("Training: {} {}".format(ger_X_trn.shape, ger_y_trn.shape))
print("Validation: {} {}".format(ger_X_val.shape, ger_y_val.shape))


# # Prepare data

# In[24]:

nclasses = 2
nfeatures = 128

ger_y_trn[ger_y_trn > 1] = 1
ger_y_val[ger_y_val > 1] = 1
ger_Y_trn = np_utils.to_categorical(ger_y_trn, nclasses)
ger_Y_val = np_utils.to_categorical(ger_y_val, nclasses)

# ## class and sample weights

# In[26]:

ger_clsw = {
    0: 1,
    1: 1.5
}
ger_splw = np.ones_like(ger_y_trn)
for c, w in ger_clsw.items():
    ger_splw[ger_y_trn == c] = w

print("Class Weights")
print(ger_clsw)

# # TRAINING

# ## Model 1

# In[30]:

model = Sequential()

model.add(kl.InputLayer(input_shape=(nfeatures, )))
model.add(kl.BatchNormalization())

model.add(kl.Dense(512, activation="sigmoid"))
model.add(kl.Dropout(0.3))  # Fraction to drop

model.add(kl.Dense(512, activation="sigmoid"))
model.add(kl.Dropout(0.3))  # Fraction to drop

model.add(kl.Dense(512, activation="sigmoid"))
model.add(kl.Dropout(0.2))  # Fraction to drop

model.add(kl.Dense(nclasses, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer=ko.Adagrad(),
              metrics=['categorical_accuracy'])

print(model.summary())


# In[36]:

batchsize = 4096 * 4
nepochs = 500

c = []
c.append(kc.EarlyStopping(patience=40, ))


print("//{:/<70}".format(" TRAIN "))

model.fit(ger_X_trn, ger_Y_trn,
        batch_size=batchsize, nb_epoch=nepochs,
        validation_data=(ger_X_val, ger_Y_val),
        class_weight=ger_clsw,
        verbose=1,
        shuffle=True,
        callbacks=c
        )


# In[37]:

preds = model.predict_classes(ger_X_val, verbose=0, batch_size=batchsize)
conx = confusion_matrix(ger_y_val, preds)
conx = conx.astype(np.float) / conx.sum(axis=1)[:, np.newaxis]
print()
print(conx)

print()
print(model.evaluate(ger_X_val, ger_Y_val, verbose=0, batch_size=batchsize))

print()
fpreds = medfilt(preds, kernel_size=9)
fconx = confusion_matrix(ger_y_val, fpreds)
fconx = fconx.astype(np.float) / fconx.sum(axis=1)[:, np.newaxis]
print(fconx)
print()
print("TIME: {}".format(time.time() - true_time))

