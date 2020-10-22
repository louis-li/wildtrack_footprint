import tensorflow as tf
import datetime
import os, re, random, logging, logging.config, sys
import numpy as np
import pandas as pd
import cv2
import imgaug.augmenters as iaa
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix
from pathlib import Path
import multiprocessing


base = 'baseline'
image_size = {'xception': 224, 'nasnet': 299, 'inception': 299}
labels = {'African elephant':0,
          'African lion':1,
          'Amur Tiger':2,
          'Bengal Tiger':3, 
          'Black Rhino':4,
          'Bongo':5,
          'Cheetah':6,
          'Jaguar':7,
          'Leopard':8,
          'Lowland Tapir':9, 
          'Otter':10,
          'Puma':11,
          'White Rhino':12}

data_dir = 'data'
model_root_dir = 'models'
today = datetime.date.today().strftime("%Y%m%d")
model_dir = f'{model_root_dir}/{today}'

seq = iaa.Sequential([
    iaa.SomeOf((0,2),[
        iaa.Identity(),
        iaa.AverageBlur(k=((3, 5), (5, 7))),
        iaa.Rotate((-45,45)),
        iaa.Affine(scale=(0.5, 0.95)),    
        iaa.Multiply((0.50, 1.1))
        ,iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=False, cval=0)
        ,iaa.Affine(shear=(-48, 48))
        ,iaa.Affine(translate_px={"x": (-42, 42), "y": (-36, 36)})
        ,iaa.KeepSizeByResize(iaa.Resize({"height": (0.70, 0.90), "width": (0.70, 0.90)}))
        ,iaa.CropAndPad(percent=(-0.2, 0.2))
        ,iaa.PerspectiveTransform(scale=(0.01, 0.1))
       ])], random_order=True)

def get_recent_model_dir(model_root_dir):
    import os, glob
    from pathlib import Path
    model_root_dir="models"
    paths = sorted(Path(model_root_dir).iterdir(), key=os.path.getmtime)[::-1]
    for p in paths:
        tflite_files = glob.glob(os.path.join(p,"*.tflite"))
        if len(tflite_files) ==3:
            return  p
    return None

def load_images(image_dir, size, debug=False):
    global labels
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if re.search('.jpg', file.lower()):

                #print(os.path.join(root, file))

                try:
                    #load image
                    img = cv2.resize(cv2.imread(os.path.join(root,file),cv2.IMREAD_COLOR),(size, size))
                    #get labels
                    dir_list = root.split('/')
                    species = dir_list[2]
                    if dir_list[1] == 'val':
                        val_x.append(img)
                        val_y.append(labels[species])
                    elif dir_list[1] != 'nolabel':
                        train_x.append(img)
                        train_y.append(labels[species])
                except:
                    print(f"Failed to load {os.path.join(root,file)}")
    
    return np.array(train_x), np.array(train_y), np.array(val_x), np.array(val_y)

def generator(features, labels, batch_size):
    while True:
        # Fill arrays of batch size with augmented data taken randomly from full passed arrays
        indexes = random.sample(range(len(features)), batch_size)
      
        # Transform X and y
        x_aug = seq(images =features[indexes])
        yield np.array(x_aug), labels[indexes]

def retrain(model_name, batch_size = 64, model_dir = 'models'):
    global recent_model_dir, image_size, data_dir
    
    # Load training & validation images
    size = image_size[model_name]
    logging.info(f"===== Loading image data for model {model_name} =====")
    train_x, train_y, val_x, val_y = load_images(f"{data_dir}", size)
    logging.info(f"Total training samples: {train_x.shape[0]}")
    logging.info(f"Total validation samples: {val_x.shape[0]}")
    
    model = tf.keras.models.load_model(f'{recent_model_dir}/{model_name}')
    adam = optimizers.Adam(learning_rate=0.0001)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                               factor=0.2,  
                               patience=3, 
                               min_lr=5e-6)
    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        min_delta=0, 
        patience=6, 
        verbose=0, 
        mode='auto',
        baseline=None, 
        restore_best_weights=True
    )

    model.compile(optimizer=adam, 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(generator(train_x, train_y, batch_size),  
                        shuffle=True,  
                        validation_data = (val_x, val_y),
                        callbacks = [reduce_lr,early_stop],                        
                        epochs=100,
                        steps_per_epoch=len(train_x)/batch_size ,
                        verbose=True)
    logging.info("Training history:")
    logging.info(history.history)
    
    
    #Get confusion matrix
    val_y_pred = np.argmax(model.predict(val_x), axis=1)
    mat = confusion_matrix(val_y, val_y_pred)
    logging.info("Confusion matrix:")
    print(mat)
    
    acc = max(history.history['val_accuracy'])
    logging.info(f"Final accuracy:{acc}")
    
    # Save trained model
    model.save(f"{model_dir}/{model_name}")
    
    # Prepare TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(f'{model_dir}/{model_name}.tflite', 'wb') as f:
        f.write(tflite_model)
        
def train_async(args):
    p = multiprocessing.Process(target=retrain, args=args)
    p.start()
    p.join()

#main
if not os.path.exists(model_dir):
    os.mkdir(model_dir) 


formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_name = f"{model_dir}/train.log"
logging.basicConfig(filename = log_name, format=formatter, level = logging.DEBUG)

# Redirect default output to log file
sys.stdout = open(log_name, "a")
    
logging.info(f"Retrain started at {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


recent_model_dir = get_recent_model_dir(model_dir)
logging.info(f"Found recent model dir:{recent_model_dir}")

train_async(args = ("xception",64,model_dir))
train_async(args = ("inception",32,model_dir))
train_async(args = ("nasnet",16,model_dir))