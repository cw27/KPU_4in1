import os # 디렉터리 경로, 파일을 활용을 운영체제가 지원하는 모듈을 이용한다.

import numpy as np 
# 파이썬으로 진행되는 데이터 분석과 인공지능 학습을 위해 numpy 도구를 이용한다.
고성능 계산, 일반 List에 비해 빠르고 효율적인 메모리 관리, Array 연산을 지원하므로
numpy를 사용한다. 
from PIL import Image
# 픽셀 단위, 이미지 필터 조작을 위해 PIL을 사용한다.
이미지 출력 및 불러오기
import random
#버퍼에서 다음 요소를 무작위로 선택하기 위해, 랜덤 함수 이용
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# 그래프를 만들고 이용하기 위한 모듈이다.
patches를 이용하여 2D 도형을 표현한다.
import xml.etree.ElementTree as et
#xml 파일을 사용하기 위한 모듈이다.

import tensorflow as tf #텐서플로우를 import as tf하면 그래프가 만들어진다.
from tensorflow import keras
#딥러닝을 하기 위해 keras 파이썬 라이브러리를 사용한다.

import re # 정규 표현식을 사용하기 위한 모듈
import shutil # 파일 복사와 삭제를 지원하는 함수 제공

from tqdm.notebook import tqdm
# 시간이 얼마나 남았는지 혹은 진행도를 확인하기 위해 사용한다.
from pandas import DataFrame
import pandas as pd
import time

# Train set : 모델을 학습하기 위한 dataset
# Validation set : 학습이 이미 완료된 모델을 검증하기 위한 dataset

#데이터 준비
IMG_SIZE = 224
cur_dir = os.getcwd()   # os.getcwd() : 현재 작업 디렉토리 확인
data_dir = os.path.join(cur_dir, 'C:\\Users\\zziya\\code1\\tutorial\\images\\annotated')
train_dir = os.path.join(data_dir, 'train\\1_xml')  # os.path.join() : 파일명과 경로를 합침
train_img_files = sorted([imgname for imgname in os.listdir(train_dir) if os.path.splitext(imgname)[-1] == '.jpg'])
train_ano_files = sorted([anoname for anoname in os.listdir(train_dir) if os.path.splitext(anoname)[-1] == '.xml'])
    # sorted() : 첫 번째 매개변수로 들어온 반복가능한 데이터를 새로운 정렬된 리스트로 만들어서 반환
dataset_files = list(zip(train_img_files,train_ano_files))
random.shuffle(dataset_files)
    # shuffle() : 고정 크기의 버퍼를 유지하면서 해당 버퍼에서 다음 요소를 무작위로 선택, 결과 확인을 위해 데이터에 인덱스를 추가

#데이터 수 계산
N_TRAIN = 3 * (len(train_img_files) // 4)
N_VAL = len(train_img_files) - N_TRAIN

train_files = dataset_files[:N_TRAIN]
val_files = dataset_files[N_TRAIN:]

#tfrecord
tfr_dir = os.path.join(data_dir,'tfrecord_afterX6')
os.makedirs(tfr_dir, exist_ok = True)   # os.makedirs() : 모든 하위 폴더 생성

tfr_train_dir = os.path.join(tfr_dir,'loc_train.tfr')
tfr_val_dir =os.path.join(tfr_dir,'loc_val.tfr')

# TFRecordWriter() : 데이터를 읽어 파일에 write
writer_train = tf.io.TFRecordWriter(tfr_train_dir)
writer_val = tf.io.TFRecordWriter(tfr_val_dir)

# string, byte 타입으로부터 BytesList 리턴
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# float(32), double(64) 타입으로부터 FloatList 리턴
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# bool, enum, int32, int64, uint64 타입으로부터 Int64List 리턴
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

for img_file, ano_file in tqdm(train_files):
    ano_path = os.path.join(train_dir, ano_file)    # os.path.join() : 파일명과 경로를 합침
    tree = et.parse(ano_path)   # parse() : 객체를 변환
    root = tree.getroot()   # getroot() : 해당 트리의 root를 반환
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
    bimage = image.tobytes()    # 바이트 객체로 이미지 반환, encoder = "raw"가 디폴트

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
        # SerializeToString() 메서드를 이용하여 문자열로 직렬화함

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
    bimage = image.tobytes()    # 바이트 객체로 이미지 반환

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
    # TFRecordDataset() : 생성된 TFRecord 파일을 불러와서 모델 학습 및 추론에 사용하기 위함
train_dataset = train_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset.map() : 입력 데이터셋의 각 원소에 주어진 함수를 적용하여 새로운 데이터셋을 생성
train_dataset = train_dataset.shuffle(buffer_size=N_TRAIN).prefetch(tf.data.experimental.AUTOTUNE).batch(N_BATCH).repeat()
    # dataset.shuffle() : 고정 크기의 버퍼를 유지하면서 해당 버퍼에서 다음 요소를 무작위로 선택, 결과 확인

val_dataset = tf.data.TFRecordDataset(tfr_val_dir)
    # TFRecordDataset() : 생성된 TFRecord 파일을 불러와서 모델 학습 및 추론에 사용하기 위함
val_dataset = val_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset.map() : 입력 데이터셋의 각 원소에 주어진 함수를 적용하여 새로운 데이터셋을 생성
val_dataset = val_dataset.batch(N_BATCH).repeat()
    # dataset.batch() : batch 사이즈를 ()로 설정. batch가 없으면 리스트에 담기지 않고 데이터를 하나씩 불러옴.

#모델 생성
def create_model():
    model = tf.keras.Sequential()   # keras.Sequential() : 레이어에 스택을 순서대로 만들어서 적용하는 모델
# 신경망 층을 구성하기위해 Sequential()를 사용한다.
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',padding='SAME', input_shape=(IMG_SIZE,IMG_SIZE,3)))
    #컨볼루션 레이어 (Conv2D)
# 필터로 입력 데이터의 이미지의 특징을 추출하고 한 칸 이동하며 연산한다.
Kenrel_size를 통해 맵의 크기를 조절한다. 
필터의 가로와 세로의 길이가 같을 때  정수 하나만 입력한다. 
레이어 중첩 사용의 경우 필터로 인해 이미지의 결과가 작아지는 것을 방지하기 위해 사용하며 Valid 혹은 same 값을 받음.
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

# 오차 함수
def loss_fn(y_true,y_pred):
    return keras.losses.MeanSquaredError()(y_true,y_pred)
        # keras.losses.MeanSquaredError() : 레이블과 예측 사이의 오차의 제곱 평균을 계산

model = create_model()
model.summary()

# 모델 학습
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=steps_per_epoch*2, decay_rate=0.5, staircase=True)
    # keras.optimizers.schedules.ExponentialDecay() : 지수 붕괴 스케줄에 사용하는 LaerningRateSchedule
model.compile(keras.optimizers.RMSprop(lr_schedule),loss=loss_fn, metrics = ['accuracy'], run_eagerly=True)
    # keras.optimizers.RMSprop() : RMSprop 알고리즘을 수현하는 최적화 프로그램
        # RMSprop은 기울기를 단순 누적하지 않고 지수 가중 이동 평균을 사용하여 최신 기울기들이 더 크게 반영되도록 함

start = time.time()

hist = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=N_EPOCHS, validation_data=val_dataset, validation_steps=validation_step)

end = time.time()

# 테스트 데이터셋 생성
test_dir = os.path.join(data_dir, 'test')
test_img_files = sorted([imgname for imgname in os.listdir(test_dir) if os.path.splitext(imgname)[-1] == '.jpg'])
test_ano_files = sorted([anoname for anoname in os.listdir(test_dir) if os.path.splitext(anoname)[-1] == '.xml'])
    # sorted() : 첫 번째 매개변수로 들어온 반복가능한 데이터를 새로운 정렬된 리스트로 만들어서 반환
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

# 모델 평가
loss_and_metrics = model.evaluate(test_x, test_y, batch_size=N_BATCH)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

# 모델 저장
# 학습된 모델들을 구분하기 위한 키
filenamekey = int(time.time())
model.save('./model/Localization_Model{}'.format(filenamekey))
model.save_weights('./model/Localization_Model_weights{}.h5'.format(filenamekey))

# 모델 요약 저장
with open('./results/modelsummary_L_{}.txt'.format(filenamekey), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# 학습 과정 시각화
plt.plot(hist.history['accuracy'], 'b', label='train accuracy')
plt.plot(hist.history['val_accuracy'], 'g', label='val accuracy')
plt.ylabel('accuracy')
plt.legend(loc='upper left')
plt.text(N_EPOCHS-2,loss_and_metrics[1], 'ta : {}'.format(format(loss_and_metrics[1],".2f")), fontsize=11, bbox=dict(boxstyle='square', color='lightgray'))
#plt.savefig('/content/gdrive/MyDrive/fire detection/result/graph/figure_L_{}.png'.format(filenamekey))
plt.savefig('./results/graph/figure_L_{}.png'.format(filenamekey))

# IOU 계산
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

# 학습결과 저장
# 학습 결과 저장용 데이터프레임 생성
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
