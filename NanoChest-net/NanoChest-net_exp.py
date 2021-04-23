# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:24:34 2020

@author: Eduardo
"""

import tensorflow as tf
import numpy as np
import os
import random
import utilities as ut


# %% Seed for repeatability
os.environ['PYTHONHASHSEED'] = str(2020)
np.random.seed(2020)
random.seed(2020)
tf.random.set_seed(2020)
# %% General configuration of the experiment. More hyperparameters and configuration is taking place later.
"""Different configurations are available.
    0 : binary configuration (using two logits with Sigmoid activation)
    1 : categorical configuration (using two logits with Softmax activation)
    
  # Arguments
    n_cfg: Number of configuration to select.
    epochs: Number of epochs to run the model.
    batch_size: Number of samples per batch.
    validation: Dir to use validation set of physical directory, IDG for Image data generator split, None for nothing.
    Trainable: Tuple (True/False, start, end)  sets the number of layers that will not be trainable, from start to end. Default configuration is True for all of the layers (True, 0, 0)
    Pooling: GAP for Global Average Pooling 2D, GMP for Global Max Pooling, MP for Max Pooling 2D
    FC_layers: A list with number of neurons for each dense layers for the Fully Connected Classifier (MLP). None is set by default.
    class_w: Penalization for imbalanced data, 'balanced' for penalization, none default, 'manual' for manual weights.
"""
cfg = ut.get_config(0, 5, 4, (True, 0, 0), 'GAP', None, None)

# Please check function get_path to use your datasets
""" Available datasets
    0  :  Pneumonia children(PNEUMO-V3) - resized 500
    1  :  COV-PNEUMO: resized 500
    2  :  COV-NOR: resized 500
    3  :  Shenzhen(TB): resized 500
    4  :  Montgomery(TB-MC) : Prep 500
    5  :  BCDR-D01 : resized 500
    6 :  BCDR-D02 : resized 500
"""

n_dataset = 4
# Load dataset from 'mem', 'dir', 'df'
source = 'dir'

path, path_df, zoom, v_flip, rot = ut.get_path(n_dataset, source)

# Type of model, pretrained, baseline, custom

model_type = 'custom'
pretrained = False
base_line = 'DenseNet121'
weights = None #'imagenet' # Or None

BATCH_SIZE = cfg['BATCH_SIZE']
EPOCHS = cfg['epochs']
DEPTH = 3
WIDTH, HEIGHT, color_mode = ut.get_dimensions(model_type, base_line, DEPTH)

INPUT_SHAPE = (WIDTH,HEIGHT,DEPTH)

# %% Getting image data generators
train_gen, val_gen, test_gen = ut.get_image_generators(source, path, path_df, cfg, WIDTH, HEIGHT, color_mode, zoom, v_flip, rot)

# %% Create full architecture
new_model = ut.create_model(cfg, INPUT_SHAPE, model_type, base_line, weights)

# %% Path to save checkpoints
path_save = path+'models/'
MODEL_NAME ='TB-MC_'+base_line+'_5e-4_Adam_NOSKW_Sig_test'

check_path = ut.create_folders(path_save, MODEL_NAME)

# %% Optimizer
lr = 5e-4
opti = ut.get_optimizer(lr, 'adam')
METRICS = ut.get_metrics_training(cfg)
    
# Compiling model
new_model.compile(opti, loss=cfg['loss'], metrics=METRICS)

# %% Callbacks
callbacks = ut.get_callbacks(check_path, path_save, MODEL_NAME)


# %% Weigthed training
class_weights = ut.get_weights_training(cfg, train_gen)

#%% Training
import time
print("Beginning training...")
start = time.time()

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = val_gen.n // val_gen.batch_size
    
history = new_model.fit(train_gen, steps_per_epoch=STEP_SIZE_TRAIN, epochs=EPOCHS, validation_data=val_gen, validation_steps=STEP_SIZE_VALID,
                        class_weight=class_weights, verbose = 0, callbacks=callbacks, initial_epoch=0)
    
end = time.time()
total_t = end - start
print("Total time: {}".format(total_t))
    
    
 # %%Training plot
ut.plot_training(path_save, MODEL_NAME)

# %% Already made predictions with model?
predictions_made = False

# %% Load model and make predictions
model, best_epoch, pred = ut.get_predictions(path_save, MODEL_NAME, test_gen, predictions_made)
 
# Compute class assignation
predicted = np.argmax(pred, axis=1)

# %% Confusion matrix
class_names = ['NORMAL', 'TUBERCULOSIS']
cm = ut.get_cmatrix(class_names, test_gen, predicted)

# %% AUC
roc_auc, auc_score, fpr, tpr = ut.compute_auc(test_gen, class_names, pred)

# %% Plot ROC Curve
ut.plot_roc(roc_auc, fpr, tpr)

# %% Save metadata
ut.save_metadata(path_save, MODEL_NAME, total_t, best_epoch, EPOCHS, BATCH_SIZE, STEP_SIZE_TRAIN, lr, class_weights, cm, auc_score)

