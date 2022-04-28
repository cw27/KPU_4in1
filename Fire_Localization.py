import os

import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xml.etree.ElementTree as et

import tensorflow as tf
from tensorflow import keras
import re
import shutil

from tqdm.notebook import tqdm
from pandas import DataFrame
import pandas as pd
import time

#데이터 준비
IMG_SIZE = 224
cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'C:\\Users\\zziya\\code1\\tutorial\\images\\annotated')
train_dir = os.path.join(data_dir, 'train\\1_xml')
train_img_files = sorted([imgname for imgname in os.listdir(train_dir) if os.path.splitext(imgname)[-1] == '.jpg'])
train_ano_files = sorted([anoname for anoname in os.listdir(train_dir) if os.path.splitext(anoname)[-1] == '.xml'])
dataset_files = list(zip(train_img_files,train_ano_files))
random.shuffle(dataset_files)
#데이터 수 계산
N_TRAIN = 3*(len(train_img_files)//4)
N_VAL = len(train_img_files) - N_TRAIN

train_files = dataset_files[:N_TRAIN]
val_files = dataset_files[N_TRAIN:]

#tfrecord
tfr_dir = os.path.join(data_dir,'tfrecord_afterX6')
os.makedirs(tfr_dir, exist_ok = True)

tfr_train_dir = os.path.join(tfr_dir,'loc_train.tfr')
tfr_val_dir =os.path.join(tfr_dir,'loc_val.tfr')

writer_train = tf.io.TFRecordWriter(tfr_train_dir)
writer_val = tf.io.TFRecordWriter(tfr_val_dir)

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


for img_file, ano_file in tqdm(train_files):
    ano_path = os.path.join(train_dir, ano_file)
    tree = et.parse(ano_path)
    root = tree.getroot()
    width = float(tree.find('./size/width').text)
    height = float(tree.find('./size/height').text)
    for object in root.iter('object'):
        xmin = int(object.find('bndbox').findtext('xmin'))
        ymin = int(object.find('bndbox').findtext('ymin'))
        xmax = int(object.find('bndbox').findtext('xmax'))
        ymax = int(object.find('bndbox').findtext('ymax'))
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    x = xc / width
    y = yc / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height

    img_path = os.path.join(train_dir, img_file)
    image = Image.open(img_path)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    bimage = image.tobytes()

    cls_num = 1

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(bimage),
        'cls_num': _int64_feature(cls_num),
        'x': _float_feature(x),
        'y': _float_feature(y),
        'w': _float_feature(w),
        'h': _float_feature(h)
    }))

    writer_train.write(example.SerializeToString())

writer_train.close()

for img_file, ano_file in tqdm(val_files):
    ano_path = os.path.join(train_dir, ano_file)
    tree = et.parse(ano_path)
    width = float(tree.find('./size/width').text)
    height = float(tree.find('./size/height').text)
    for object in root.iter('object'):
        xmin = int(object.find('bndbox').findtext('xmin'))
        ymin = int(object.find('bndbox').findtext('ymin'))
        xmax = int(object.find('bndbox').findtext('xmax'))
        ymax = int(object.find('bndbox').findtext('ymax'))
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    x = xc / width
    y = yc / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height

    img_path = os.path.join(train_dir, img_file)
    image = Image.open(img_path)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    bimage = image.tobytes()

    cls_num = 1

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(bimage),
        'cls_num': _int64_feature(cls_num),
        'x': _float_feature(x),
        'y': _float_feature(y),
        'w': _float_feature(w),
        'h': _float_feature(h)
    }))

    writer_val.write(example.SerializeToString())

writer_val.close()

#하이퍼파라미터 설정
N_EPOCHS = 10
N_BATCH = 16
IMG_SIZE = 224
learning_rate = 0.001
steps_per_epoch = int(N_TRAIN/N_BATCH)
validation_step = int(np.ceil(N_VAL / N_BATCH))

#데이터셋 생성(TFrecord 읽기)
def _parse_function(tfrecord_serialized):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'cls_num': tf.io.FixedLenFeature([], tf.int64),
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        'w': tf.io.FixedLenFeature([], tf.float32),
        'h': tf.io.FixedLenFeature([], tf.float32)
    }

    parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)

    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])
    image = tf.cast(image, tf.float32) / 255.

    cls_label = tf.cast(parsed_features['cls_num'], tf.int64)

    x = tf.cast(parsed_features['x'], tf.float32)
    y = tf.cast(parsed_features['y'], tf.float32)
    w = tf.cast(parsed_features['w'], tf.float32)
    h = tf.cast(parsed_features['h'], tf.float32)
    gt = tf.stack([x, y, w, h], -1)

    return image, gt


train_dataset = tf.data.TFRecordDataset(tfr_train_dir)
train_dataset = train_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=N_TRAIN).prefetch(tf.data.experimental.AUTOTUNE).batch(
    N_BATCH).repeat()

val_dataset = tf.data.TFRecordDataset(tfr_val_dir)
val_dataset = val_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(N_BATCH).repeat()

#모델 생성
def create_model():
    model = tf.keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',padding='SAME', input_shape=(IMG_SIZE,IMG_SIZE,3)))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu',padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu',padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(4,activation='sigmoid'))
    return model

def loss_fn(y_true,y_pred):
    return keras.losses.MeanSquaredError()(y_true,y_pred)

model = create_model()
model.summary()

#학습
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                         decay_steps=steps_per_epoch*2,
                                                         decay_rate=0.5,
                                                         staircase=True)
model.compile(keras.optimizers.RMSprop(lr_schedule),loss=loss_fn, metrics = ['accuracy'], run_eagerly=True)


start = time.time()

hist = model.fit(train_dataset, steps_per_epoch=steps_per_epoch,
         epochs=N_EPOCHS,
         validation_data=val_dataset,
         validation_steps=validation_step)

end = time.time()

#테스트 데이터셋 생성
test_dir = os.path.join(data_dir, 'test')
test_img_files = sorted([imgname for imgname in os.listdir(test_dir) if os.path.splitext(imgname)[-1] == '.jpg'])
test_ano_files = sorted([anoname for anoname in os.listdir(test_dir) if os.path.splitext(anoname)[-1] == '.xml'])
test_dataset_files = list(zip(test_img_files, test_ano_files))

test_x = []
test_y = []
for img_file, ano_file in tqdm(test_dataset_files):
    ano_path = os.path.join(test_dir, ano_file)
    tree = et.parse(ano_path)
    width = float(tree.find('./size/width').text)
    height = float(tree.find('./size/height').text)
    xmin = float(tree.find('./object/bndbox/xmin').text)
    ymin = float(tree.find('./object/bndbox/ymin').text)
    xmax = float(tree.find('./object/bndbox/xmax').text)
    ymax = float(tree.find('./object/bndbox/ymax').text)
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    x = xc / width
    y = yc / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height

    img_path = os.path.join(test_dir, img_file)
    image = Image.open(img_path)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = tf.cast(np.array(image), tf.float32) / 255.
    test_x.append(image)
    test_y.append(np.array([x, y, w, h]))

test_x = np.array(test_x)
test_y = np.array(test_y)

#모델 평가
loss_and_metrics = model.evaluate(test_x, test_y, batch_size=N_BATCH)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

#모델 저장
#학습된 모델들을 구분하기 위한 키
filenamekey = int(time.time())
model.save('./model/Localization_Model{}'.format(filenamekey))
model.save_weights('./model/Localization_Model_weights{}.h5'.format(filenamekey))
#모델 요약 저장
with open('./results/modelsummary_L_{}.txt'.format(filenamekey), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

#학습 과정 시각화
plt.plot(hist.history['accuracy'], 'b', label='train accuracy')
plt.plot(hist.history['val_accuracy'], 'g', label='val accuracy')
plt.ylabel('accuracy')
plt.legend(loc='upper left')
plt.text(N_EPOCHS-2,loss_and_metrics[1], 'ta : {}'.format(format(loss_and_metrics[1],".2f")), fontsize=11, bbox=dict(boxstyle='square', color='lightgray'))
#plt.savefig('/content/gdrive/MyDrive/fire detection/result/graph/figure_L_{}.png'.format(filenamekey))
plt.savefig('./results/graph/figure_L_{}.png'.format(filenamekey))


#IOU 계산
test_dataset = list(zip(test_x, test_y))
N_TEST = len(test_dataset)
avg_iou = 0
num_imgs = N_TEST
for i, (test_data, test_gt) in enumerate(test_dataset):
    flag = (i == N_TEST - 1)
    x = test_gt[0]
    y = test_gt[1]
    w = test_gt[2]
    h = test_gt[3]
    test_data = test_data[tf.newaxis, ...]
    prediction = model.predict(test_data)
    pred_x = prediction[0][0]
    pred_y = prediction[0][1]
    pred_w = prediction[0][2]
    pred_h = prediction[0][3]

    xmin = int((x - w / 2.) * IMG_SIZE)
    ymin = int((y - h / 2.) * IMG_SIZE)
    xmax = int((x + w / 2.) * IMG_SIZE)
    ymax = int((y + h / 2.) * IMG_SIZE)

    pred_xmin = int((pred_x - pred_w / 2.) * IMG_SIZE)
    pred_ymin = int((pred_y - pred_h / 2.) * IMG_SIZE)
    pred_xmax = int((pred_x + pred_w / 2.) * IMG_SIZE)
    pred_ymax = int((pred_y + pred_h / 2.) * IMG_SIZE)
    if xmin > pred_xmax or xmax < pred_xmin or ymin > pred_ymax or ymax < pred_ymin:
        continue
    w_union = np.max((xmax, pred_xmax)) - np.min((xmin, pred_xmin))
    h_union = np.max((ymax, pred_ymax)) - np.min((ymin, pred_ymin))
    w_inter = np.min((xmax, pred_xmax)) - np.max((xmin, pred_xmin))
    h_inter = np.min((ymax, pred_ymax)) - np.max((ymin, pred_ymin))

    w_sub1 = np.abs(xmax - pred_xmax)
    h_sub1 = np.abs(ymax - pred_ymax)
    w_sub2 = np.abs(xmin - pred_xmin)
    h_sub2 = np.abs(ymin - pred_ymin)

    iou = (w_inter * h_inter) / ((w_union * h_union) - (w_sub1 * h_sub1) - (w_sub2 * h_sub2))
    avg_iou += iou
avg_iou = avg_iou / N_TEST

#학습결과 저장
#학습 결과 저장용 데이터프레임 생성
data = DataFrame(columns=['key','train data len',
                          'validation data len',
                          'EPOCHS',
                          'learning_rate_schedule',
                          'batch size',
                          'loss_func',
                          'optimizer',
                          'comment',
                          'test_accuracy',
                          'MeanIOU'
                          'train time'
                         ])

data = pd.read_csv('./results/HyperParameter_L.csv',encoding='euc-kr')
data = data.append({'key':filenamekey,
                    'train data len': N_TRAIN,
                    "validation data len":N_VAL,
                    "EPOCHS":N_EPOCHS,
                    "learning_rate_schedule":"ExponentialDecay(initial_learning_rate={},decay_steps={},decay_rate=0.5,staircase=True)".format(learning_rate,steps_per_epoch*2),
                    "batch size":N_BATCH,
                    "loss_func":'MSE',
                    "optimizer":"RMSprop",
                    "comment": "초기 모델 + 데이터변경 + 데이터증강 + 배치정규화 + Dropout 0.4->0.2 + 출력층sigmoid",
                    "test_accuracy":loss_and_metrics[1],
                    "MeanIOU":avg_iou,
                    "train time":format(end-start, ".2f")},ignore_index=True)

#data.to_csv('/content/gdrive/MyDrive/fire detection/result/HyperParameter_L.csv',encoding='euc-kr')
data.to_csv('./results/HyperParameter_L.csv',encoding='euc-kr',index=False)