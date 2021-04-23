# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:57:36 2021

@author: Eduardo
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, SeparableConv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Dropout, Flatten
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import os
import glob
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc as auc_roc
from csv import DictWriter
from datetime import date

import seaborn as sns

# Utilities for experiment on NanoChest-net

def NanoChestnet(INPUT_SHAPE):
    
    input_img = Input(shape=INPUT_SHAPE)
    
    x = Conv2D(64, (3,3),  dilation_rate=(2,2))(input_img)
    x = relu(x)
    
    x = Conv2D(64, (3,3),  dilation_rate=(2,2))(input_img)
    x = relu(x)
    
    x = MaxPooling2D((3,3))(x)
    
    x = SeparableConv2D(128, (3,3), depth_multiplier=3, dilation_rate=(2,2))(x)
    x = BatchNormalization()(x)
    x = relu(x)
       
    x = SeparableConv2D(256, (3,3), depth_multiplier=3, dilation_rate=(2,2))(x)
    x = BatchNormalization()(x)
    x = relu(x)
    
    x = MaxPooling2D((3,3))(x)
    
    x = SeparableConv2D(256, (3,3), depth_multiplier=3, dilation_rate=(2,2))(x)
    x = BatchNormalization()(x)
    x = relu(x)
       
    x = SeparableConv2D(512, (3,3), depth_multiplier=3, dilation_rate=(2,2))(x)
    x = BatchNormalization()(x)
    x = relu(x)
    
    
    
    x = SeparableConv2D(1024, (3,3), strides=1, dilation_rate=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = relu(x)
    
    x = SeparableConv2D(2048, (3,3), strides=1, dilation_rate=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = relu(x)
    
    
    gap = GlobalAveragePooling2D()(x)
    drop1 = Dropout(0.25)(gap)
        
    output = Dense(2, activation='softmax')(drop1)
            
    model = Model(inputs=input_img, outputs=output)
      
    return model

def get_config(n_cfg, epochs, batch_size=16, trainable=(True, 0, 0), pooling='GAP', FC_layers=None, classw=None):
  
  cfg = []
  hparams_binary = {
            "class_mode": "binary",
            "BATCH_SIZE": batch_size,
            "trainable": trainable,
            "epochs": epochs,
            "pooling": pooling,
            "FC_LAYERS" : FC_layers,
            "dropout": 0.25,
            "logits": 2,
            "activation" : 'sigmoid',
            "loss": 'binary_crossentropy',
            "classw": classw
            }
  hparams_categorical = {
            "class_mode": "categorical",
            "BATCH_SIZE": batch_size,
            "trainable": trainable,
            "epochs": epochs,
            "pooling": pooling,
            "FC_LAYERS" : FC_layers,
            "dropout": 0.25,
            "logits": 2,
            "activation" : 'softmax',
            "loss": 'categorical_crossentropy',
            "classw": classw
            }
  cfg.append(hparams_binary)
  cfg.append(hparams_categorical)
  
  return  cfg[n_cfg]


## Change for your own paths
def get_path(n_dataset, source):
    
    datasets = ['Pneumonia-children/V3/resized/500/',
                'COVID19-IEEE/COV-PNEUMO/resized/500/' ,'COVID19-IEEE/COV-NOR/resized/500/' ,
                'Tuberculosis/Shenzhen/resized/500/','Tuberculosis/Montgomery/Prep/500/',
                'BCDR/BCDR-D01/Original-jpg/resized/500/',
                'BCDR/BCDR-D02/resized/500/']
                
    path = 'E:/CS/Datasets/'+datasets[n_dataset]
    
    # Name of csv containing name files for mammography datasets
    df_files = ['metadata_bcdr-d01.csv', 'metadata_bcdr-d02.csv']
                
    zoom = [0.9,1.2]
    
    if source == 'df':
        if n_dataset == 5:
            path_df = path+df_files[0]
        elif n_dataset == 6:
            path_df = path+df_files[1]
        v_flip = True
        rot = 30
        
    else:
        if n_dataset == 4:
            zoom = [0.9,1.5]
            
        path_df = None
        v_flip = False
        rot = 20
                
    return path, path_df, zoom, v_flip, rot


def image_generators(path, WIDTH, HEIGHT, BATCH_SIZE, class_mode='categorical', color_mode='grayscale', mode='dir', df_train=None, df_val=None, df_test=None, val_split=None, zoom=[0.90,1.2], v_flip=False, rot=20):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=zoom, #0.9-1.2 for all datasets, excepto for Montgomery 0.9-1.5.
        horizontal_flip=True,
        vertical_flip=v_flip, #True only for Mammography datasets
        width_shift_range=0.20, 
        height_shift_range=0.20,
        rotation_range=rot, # 20 Shenzhen, Montgomery, Pneumonia, COVID variants. 30 for Mammography datasets.
        brightness_range = [0.80,1.05],
        fill_mode='constant', 
        cval=0,
        validation_split = val_split
        )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    if mode == 'dir':
        train_generator = train_datagen.flow_from_directory(
            path+'training/',
            target_size=(WIDTH, HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode=class_mode,color_mode=color_mode,
            seed = 2020)
        
        validation_generator = test_datagen.flow_from_directory(
            path+'validation/',
            target_size=(WIDTH, HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode=class_mode,color_mode=color_mode,
            seed = 2020)
        
        test_generator = test_datagen.flow_from_directory(
            path+'testing/',
            target_size=(WIDTH, HEIGHT),
            batch_size=1,
            color_mode=color_mode,
            class_mode=None,
            shuffle=False
            )
    elif mode == 'df':
        train_generator = train_datagen.flow_from_dataframe(
           df_train,
           directory=path+'images/',
           x_col="filename",
           y_col="class",
           weight_col=None,
           target_size=(WIDTH, HEIGHT),
           color_mode=color_mode,
           class_mode=class_mode,
           batch_size=BATCH_SIZE,
           seed=2020,
           subset=None,
           shuffle=True,
           validate_filenames=True,
           )

        validation_generator = test_datagen.flow_from_dataframe(
            df_val,
            directory=path+'images/',
            x_col="filename",
            y_col="class",
            weight_col=None,
            target_size=(WIDTH, HEIGHT),
            color_mode=color_mode,
            class_mode=class_mode,
            batch_size=BATCH_SIZE,
            seed=2020,
            subset=None,
            shuffle=True,
            validate_filenames=True,
            )
        
        test_generator = test_datagen.flow_from_dataframe(
            df_test,
            directory=path+'images/',
            x_col="filename",
            y_col="class",
            weight_col=None,
            target_size=(WIDTH, HEIGHT),
            color_mode=color_mode,
            class_mode=class_mode,
            batch_size=1,
            seed=2020,
            subset=None,
            shuffle=False,
            validate_filenames=True,
            )
            
        
    return train_generator, validation_generator, test_generator

def get_dimensions(model_type, base_line, DEPTH):
    if model_type == 'baseline' or model_type == 'pretrained':
        if base_line == 'Xception':
            WIDTH = HEIGHT = 299
        elif base_line == 'ResNet50' or base_line == 'DenseNet121' or base_line == 'VGG16':
            WIDTH = HEIGHT = 224
      
    elif model_type == 'custom':
        WIDTH = HEIGHT = 250
    
    if DEPTH == 3:
        color_mode = 'rgb'
    else:
        color_mode = 'grayscale'
        
    return WIDTH, HEIGHT, color_mode

# Create image generator for all subsets
def get_image_generators(source, path, path_df, cfg, WIDTH, HEIGHT, color_mode, zoom, v_flip, rot):
    if source == 'dir':
    # Using data generator from directory
        train_gen, val_gen, test_gen = image_generators(path, WIDTH, HEIGHT, cfg['BATCH_SIZE'], 'categorical', color_mode, val_split=None, zoom=zoom, v_flip=v_flip, rot=rot)
        cfg['logits'] = len(np.unique(train_gen.classes))
        
    elif source == 'df':
    # Data generator from dataframe
        full_data = pd.read_csv(path_df)
        full_data = full_data.sample(frac=1, random_state=2020)
        partitions = [.70,.10,.20]
        split_1 = int(partitions[0] * len(full_data))
        split_2 = int(partitions[1] * len(full_data))
    
        train = full_data[:split_1]
        val = full_data[split_1:split_1+split_2]
        test = full_data[split_1+split_2:]
    
        train_gen, val_gen, test_gen = image_generators(path, WIDTH, HEIGHT, cfg['BATCH_SIZE'],'categorical', color_mode, mode='df', df_train=train, df_val=val, df_test=test, val_split=None, zoom=zoom, v_flip=v_flip, rot=rot)
    
    return train_gen, val_gen, test_gen

def load_full_dataset(path):
    


    dataset = np.load(path)
    X_train = dataset['x_train']
    Y_train = dataset['y_train']
    X_val = dataset['x_val']
    Y_val = dataset['y_val']
    X_test = dataset['x_test']
    Y_test = dataset['y_test']

    return X_train, Y_train, X_val, Y_val, X_test, Y_test
        
def get_label(path, CLASSES_NAME):
    parts =  path.split('\\')
    l = parts[-2] == CLASSES_NAME
    label = np.argwhere(l)
    return label

def load_set(DIR, INPUT_SHAPE):
    
    CLASSES_NAME = np.array(os.listdir(DIR))
    filenames = glob(DIR+"*\\*")
    
    x = []
    y = []
    for ix, i in enumerate(filenames):
        im = np.array(cv2.imread(i))
        im = np.reshape(im, INPUT_SHAPE)
        x.append(im)
        y.append(get_label(i, CLASSES_NAME))
        #print(ix)

    
    X = np.array(x)
    Y = y
    
    
    return X, Y        


# Normalization or standarization when loading from memory 
def normalization(X_train, X_val, X_test):
    X_train = X_train.astype('float32') 
    X_val = X_val.astype('float32') 
    X_test = X_test.astype('float32') 

    #if norm:
    #  X_train /= 255.
    #  X_val /= 255.
    #  X_test /= 255.
    #else:
    # Subtract pixel mean
    x_train_mean = np.mean(X_train, axis=0)
    X_train -= x_train_mean
    X_val -= x_train_mean
    X_test -= x_train_mean
    # Divide by std dev
    x_train_std = np.std(X_train, axis=0)
    X_train /= x_train_std
    X_val /= x_train_std
    X_test /= x_train_std
    
    return X_train, X_val, X_test
    
def cat_classes(Y_train, Y_val, Y_test):
    Y_trainc = to_categorical(Y_train, num_classes=2)
    Y_valc = to_categorical(Y_val, num_classes=2)
    Y_testc = to_categorical(Y_test, num_classes=2)
    
    return Y_trainc, Y_valc, Y_testc

def build_finetuning_model(cfg, base_model, model_type='imagenet'):
    
    for layer in base_model.layers:
      layer.trainable = True

    if cfg["trainable"][0] == False:
      
      for layer in base_model.layers[cfg["trainable"][1]:cfg["trainable"][2]]:
        layer.trainable = False
        
    if model_type == 'imagenet':
        x = base_model.output
    elif model_type == 'pretrained' or model_type == 'custom':   
        x = base_model.layers[-4].output
   
    if cfg["pooling"] == 'GAP':
      x =  GlobalAveragePooling2D()(x)
    elif cfg["pooling"] == 'GMP':
      x =  GlobalMaxPooling2D()(x)
    elif cfg["pooling"] == 'MP':
      x =  MaxPooling2D()(x)
      x = Flatten()(x)

    x = Dropout(cfg["dropout"])(x)
    
    if not cfg["FC_LAYERS"] == None:
      for fc in cfg["FC_LAYERS"]:
        x = Dense(fc, activation='relu')(x)
        x = Dropout(0.25)(x)
        
    #New output layer
    output = Dense(cfg["logits"], activation = cfg["activation"])(x)
        
    finetune_model = Model(inputs=[base_model.input], outputs=[output])
    
    return finetune_model

def create_model(cfg, INPUT_SHAPE, model_type, base_line, weights):

    if model_type == 'custom':
        base_model = NanoChestnet(INPUT_SHAPE)
        new_model =  build_finetuning_model(cfg, base_model, model_type) 
        base_line = 'NanoChest'
    elif model_type == 'baseline':
        # Finetuning pretrained model on imagenet
        if base_line == 'Xception':
            base_model = tf.keras.applications.Xception(weights=weights, include_top=False, input_shape=INPUT_SHAPE, pooling=None, classes=1000)
        elif base_line == 'ResNet50':
            base_model = tf.keras.applications.ResNet50(weights=weights, include_top=False, input_shape=INPUT_SHAPE, pooling=None, classes=1000)
        elif base_line == 'DenseNet121':
            base_model = tf.keras.applications.DenseNet121(weights=weights, include_top=False, input_shape=INPUT_SHAPE, pooling=None, classes=1000)
        elif base_line == 'VGG16':
            base_model = tf.keras.applications.VGG16(weights=weights, include_top=False, input_shape=INPUT_SHAPE, pooling=None, classes=1000)
        
        new_model = build_finetuning_model(cfg, base_model)    
        
    return new_model

# Create folder to save trained models
def create_folders(path_save, MODEL_NAME):
    try:
        # Create target Directory
        os.makedirs(path_save+MODEL_NAME+'/')    
        print('Directory ' , MODEL_NAME ,  ' Created ') 
    except FileExistsError:
        print('Directory ' , MODEL_NAME ,  ' already exists')
    
    
    #File path for checkpoints
    check_path=path_save+MODEL_NAME+"/"+"best_model.{epoch:02d}.h5"
    
    return check_path

def get_optimizer(lr, opt='adam'):
    
    if opt == 'adam':
        opti = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    elif opt == 'sgd':
        opti = SGD(lr=lr)
    elif opt == 'rms':
        opti = RMSprop(lr=lr)
        
    return opti

def get_metrics_training(cfg):
    METRICS = [ tf.metrics.Precision(name='precision'), tf.metrics.Recall(name='recall'), tf.metrics.AUC(name='auc')]
    if cfg['class_mode'] == 'binary':
        METRICS.append(tf.metrics.BinaryAccuracy(name='accuracy'))
    else:
        METRICS.append(tf.metrics.CategoricalAccuracy(name='accuracy'))
        
    return METRICS

# Create callbacks for checkpoint and logger
def get_callbacks(check_path, path_save, MODEL_NAME):
    
    # checkpoint callback
    checkpoint = ModelCheckpoint(check_path, monitor='val_loss', verbose=1, mode='min', save_best_only=True, save_freq='epoch')
    # CSVLogger to register all metrics during training time
    csv_logger = CSVLogger(path_save+MODEL_NAME+"/model_history_log.csv", append=True)
    
    callbacks_list = [checkpoint, csv_logger]
    
    return callbacks_list

# Weighted training if needed
def get_weights_training(cfg, train_gen):
    if cfg['classw'] == 'balanced':
        Y_train = np.array(train_gen.classes)
        class_w = class_weight.compute_class_weight('balanced',
                                                          np.unique(Y_train),
                                                          np.reshape(Y_train,(len(Y_train),)))
        class_weights = {0:class_w[0], 1: class_w[1] }        
      
    elif cfg['classw'] == None:
        class_weights = None
        
    return class_weights

# Change the name of your csv if needed
def plot_training(path_save, MODEL_NAME):
    
    df = pd.read_csv(path_save+MODEL_NAME+'/model_history_log.csv')
    print_metric(df['loss'], df['val_loss'], 'Model loss', 'Epoch', 'Loss', ['Loss','Validation loss'])
    print_metric(df['accuracy'], df['val_accuracy'], 'Model accuracy', 'Epoch', 'Accuracy', ['Accuracy','Validation Accuracy'])
    print_metric(df['auc'], df['val_auc'], 'Model AUC', 'Epoch', 'AUC', ['AUC','Validation AUC'])
        

def load_best_model(path, model_name):
  list_of_models = glob.glob(path+model_name+'/*.h5')
  best_model = os.path.basename(max(list_of_models, key=os.path.getctime))
  epoch =int(best_model.split('.')[1])
  model = load_model(path+model_name+'/'+best_model)
  return model, epoch

def get_predictions(path_save, MODEL_NAME, test_gen, predictions_made):
    model = None
    if predictions_made:
      model, best_epoch = load_best_model(path_save, MODEL_NAME)
      # summarize model.
     # model.summary()
      list_pred = glob.glob(path_save+MODEL_NAME+'/*.npy')
      name_pred = os.path.basename(max(list_pred, key=os.path.getctime))
      pred = np.load(path_save+MODEL_NAME+'/'+name_pred)
    else:
      model, best_epoch = load_best_model(path_save, MODEL_NAME)
      # summarize model.
      #model.summary() 
      print(best_epoch)
     
      test_gen.reset()
      pred = model.predict(test_gen, verbose=1, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
      np.save(path_save+MODEL_NAME+'/predictions'+str(best_epoch), pred)
      
      return model, best_epoch, pred

def print_metric(m1,m2,title, xlabel, ylabel, metrics_name):
    plt.figure()
    plt.plot(m1, marker='.')
    plt.plot(m2, marker='.')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(metrics_name, loc="lower right")
    plt.ylim(0,1.0)
    plt.show()

def get_cmatrix(class_names, test_gen, predicted):
    y_true = test_gen.classes
    y_pred = predicted
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(2, figsize=(4,4), dpi=300)
    
    
    df_cm = pd.DataFrame(cm, columns=class_names, index=class_names)
    df_cm.index.name = 'True label'
    df_cm.columns.name = 'Predicted label'
    
    # sn.set(font_scale=1.5) #for label size
    heatmap_cm = sns.heatmap(df_cm, annot=True, fmt='d',  cbar=False, cmap=plt.cm.Blues)
    
    sns.set(font_scale=1)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label',fontsize=12)
    plt.show()
    
    
    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return cm

def compute_auc(test_gen, class_names, pred):
    Y_testc = to_categorical(test_gen.classes, num_classes=len(class_names))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(Y_testc[:, i], pred[:, i] )
        roc_auc[i] = auc_roc(fpr[i], tpr[i])
    
    # micro calculate metrics globally considerig each element of the label indicator matrix as a label.
    #fpr, tpr, _ = roc_curve(Y_test, predicted)
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_testc.ravel(), pred.ravel())
    roc_auc["micro"] = auc_roc(fpr["micro"], tpr["micro"])
    auc_score = roc_auc_score(Y_testc[:, i], pred[:, i], average='weighted')

    return roc_auc, auc_score, fpr, tpr

def plot_roc(roc_auc, fpr, tpr):
    plt.figure(dpi=300)
    plt.plot(fpr[1], tpr[1], color='orange', marker='.',
              label='Disease (area = {0:.3f})'.format(roc_auc[1]))
    plt.plot([0, 1], [0, 1],  linestyle='--', label='No skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def get_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    specificity = tn/(tn+fp)
    
    return accuracy, precision, recall, specificity, f1


def append_dict_as_row(file_name, dict_data):
    field_names = ['1-Date','2-Model name','3-Best epoch','4-Accuracy','5-Precision','6-Recall','7-Specificity',
                   '8-F1','9-AUC','10-Epochs','11-Batch size','12-lr','13-Weights','14-Number of batches',
                   '15-Total training time (s)','16-Epoch avg time (s)','17-Time per example','18-Convergence time']
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        writer = DictWriter(write_obj, fieldnames=field_names)
        #writer.writeheader()
        writer.writerow(dict_data)
        
def append_dict_as_row_cv(file_name, dict_data):
    field_names = ['1-Date','2-Model name','3-Avg Accuracy','4-Avg Precision','5-Avg Recall',
                   '6-Avg F1','7-Avg AUC','8-k-fold','9-Epochs','10-Batch size','11-lr',
                   '12-Weights','13-Total training time (s)','14-Epoch avg time (s)']
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        writer = DictWriter(write_obj, fieldnames=field_names)
        #writer.writeheader()
        writer.writerow(dict_data)
        
def save_metadata(path_save, MODEL_NAME, total_t, best_epoch, EPOCHS, BATCH_SIZE, STEP_SIZE_TRAIN, lr, class_weights, cm, auc_score):
    epoch_time = total_t / EPOCHS
    date_exp = date.today().strftime("%d/%m/%Y")
    acc, pr, re, sp, f1 = get_metrics(cm)
    exp_data = {"1-Date": date_exp, "2-Model name": MODEL_NAME, "3-Best epoch": best_epoch,
            "4-Accuracy": acc ,"5-Precision": pr, "6-Recall": re, "7-Specificity": sp, "8-F1": f1, "9-AUC": auc_score,
            "10-Epochs": EPOCHS, "11-Batch size": BATCH_SIZE, "12-lr": lr,
            "13-Weights": str(class_weights), "14-Number of batches": STEP_SIZE_TRAIN, "15-Total training time (s)": total_t,
            "16-Epoch avg time (s)": epoch_time,  "17-Time per example": (epoch_time/(STEP_SIZE_TRAIN*BATCH_SIZE)),
            "18-Convergence time": best_epoch*epoch_time}
    
    # Write metadata to main file
    main_metadata_file = path_save+'Experiments_info.csv'
    append_dict_as_row(main_metadata_file, exp_data)