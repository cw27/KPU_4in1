import os, sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from tqdm.notebook import tqdm
from pandas import DataFrame
import pandas as pd
import time


# 이전 학습 결과 불러오기
data = DataFrame(columns=['key','common train data len','fire train data len','smoke train data len',
                          'common validation data len','fire validation data len','smoke validation data len',
                          'EPOCHS',
                          'learning_rate_schedule',
                          'batch size',
                          'loss_func',
                          'optimizer',
                          'comment',
                          'test_accuracy',
                          'train time'
                         ])
data = pd.read_csv('./results/HyperParameter.csv',encoding='euc-kr')


#데이터 준비
IMG_SIZE = 224
cur_dir = os.getcwd() # os.getcwd() : 현재 작업 디렉토리 확인
# data_dir = os.path.join(cur_dir,'images')
camera_dir = os.path.join(cur_dir,'images') # os.path.join() : 파일명과 경로를 합침
camera0_dir = os.path.join(camera_dir,'0')
camera1_dir = os.path.join(camera_dir,'1')
camera2_dir = os.path.join(camera_dir,'2')


#경로 설정
normal_img_files = os.listdir(camera0_dir)  # od.listdir() : 입력 경로 내의 모든 파일과 폴더명 리스트 반환
fire_img_files = os.listdir(camera1_dir)
smoke_img_files = os.listdir(camera2_dir)
print(len(normal_img_files), len(fire_img_files), len(smoke_img_files))
tfr_dir = os.path.join(camera_dir,'tfrecord')       # os.path.join() : 파일명과 경로를 합침
os.makedirs(tfr_dir, exist_ok=True)                 # os.makedirs() : 모든 하위 폴더 생성
train_tfr_dir = os.path.join(tfr_dir,'train.tfr')   # 학습시킬 구조화된 폴더와 이미지
val_tfr_dir = os.path.join(tfr_dir,'val.tfr')       # 평가확인할 구조화된 폴더와 이미지
tfr_dir = os.path.join(camera_dir,'tfrecord-before augmentation')
os.makedirs(tfr_dir, exist_ok=True)
train_tfr_dir = os.path.join(tfr_dir,'train.tfr')
val_tfr_dir = os.path.join(tfr_dir,'val.tfr')

N_N_TRAIN = 3 * (len(normal_img_files) // 4)
N_N_VAL = len(normal_img_files) - N_N_TRAIN
N_F_TRAIN = 3 * (len(fire_img_files) // 4)
N_F_VAL = len(fire_img_files) - N_F_TRAIN
N_S_TRAIN = 3 * (len(smoke_img_files) // 4)
N_S_VAL = len(smoke_img_files) - N_S_TRAIN

# 데이터의 양이 많은 경우 이를 Binary로 Serialization한 뒤,
# 파일 형태로 저장하고 있다가 이를 다시 읽어들이는 형태로 처리하면 속도상의 이득을 기대할 수 있기 때문에 TFRecord 사용
train_tfr_writer = tf.io.TFRecordWriter(train_tfr_dir)
val_tfr_writer = tf.io.TFRecordWriter(val_tfr_dir)

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

# enumerate는 (i, files) = (인덱스, 원소) 로 이루어진 튜플로 만들어주는 함수
for i, file in enumerate(tqdm(normal_img_files)):
    img_path = os.path.join(camera0_dir, file)      # os.path.join() : 파일명과 경로를 합침
    image = Image.open(img_path)                    # 이미지 오픈
    image = image.resize((IMG_SIZE, IMG_SIZE))      # 이미지 리사이즈
    bimage = image.tobytes()        # 바이트 객체로 이미지 반환, encoder = "raw"가 디폴트

    # value - tf.example에 적합한 타입 형태의 dictionary를 생성
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(bimage),
        'cls_num': _int64_feature(0)
    }))
    if i < N_N_TRAIN:
        train_tfr_writer.write(example.SerializeToString())
    else:
        val_tfr_writer.write(example.SerializeToString())
        # SerializeToString() 메서드를 이용하여 문자열로 직렬화함

# enumerate는 (i, files) = (인덱스, 원소) 로 이루어진 튜플로 만들어주는 함수
for i, file in enumerate(tqdm(fire_img_files)):
    img_path = os.path.join(camera1_dir, file)      # os.path.join() : 파일명과 경로를 합침
    image = Image.open(img_path)                    # 이미지 오픈
    image = image.resize((IMG_SIZE, IMG_SIZE))      # 이미지 리사이즈
    bimage = image.tobytes()        # 바이트 객체로 이미지 반환

    # value - tf.example에 적합한 타입 형태의 dictionary를 생성
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(bimage),
        'cls_num': _int64_feature(1)
    }))

    if i < N_F_TRAIN:
        train_tfr_writer.write(example.SerializeToString())
    else:
        val_tfr_writer.write(example.SerializeToString())

# enumerate는 (i, files) = (인덱스, 원소) 로 이루어진 튜플로 만들어주는 함수
for i, file in enumerate(tqdm(smoke_img_files)):
    img_path = os.path.join(camera2_dir, file)      # os.path.join() : 파일명과 경로를 합침
    image = Image.open(img_path)                    # 이미지 오픈
    image = image.resize((IMG_SIZE, IMG_SIZE))      # 이미지 리사이즈
    bimage = image.tobytes()        # 바이트 객체로 이미지 반환

    # value - tf.example에 적합한 타입 형태의 dictionary를 생성
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(bimage),
        'cls_num': _int64_feature(2)
    }))

    if i < N_S_TRAIN:
        train_tfr_writer.write(example.SerializeToString())
    else:
        val_tfr_writer.write(example.SerializeToString())

train_tfr_writer.close()
val_tfr_writer.close()


#데이터 수 계산
N_TRAIN = N_F_TRAIN+N_N_TRAIN+N_S_TRAIN #train 데이터의 총 개수
N_VAL = N_F_VAL + N_N_VAL + N_S_VAL #vlaidation 데이터의 총 개수
N_BATCH = 32 #batch size


# 디스크에서 학습 데이터를 필요한 만큼 불러오는 인스턴스 생성
def _parse_function(tfreced_serialized):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'cls_num': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_features = tf.io.parse_single_example(tfreced_serialized, features)

    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        # tf.decode_raw() : 원시 바이트 문자열을 텐서로 변환
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])
        # tf.reshape() : 텐서의 형태 변환
    image = tf.cast(image, tf.float32) / 255.
        # tf.cast() : 텐서를 새로운 형태로 캐스팅하는 데에 사용

    cls_label = tf.cast(parsed_features['cls_num'], tf.int64)

    return image, cls_label

train_dataset = tf.data.TFRecordDataset(train_tfr_dir)
    # TFRecordDataset() : 생성된 TFRecord 파일을 불러와서 모델 학습 및 추론에 사용하기 위함
train_dataset = train_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset.map() : 입력 데이터셋의 각 원소에 주어진 함수를 적용하여 새로운 데이터셋을 생성
train_dataset = train_dataset.shuffle(buffer_size=N_TRAIN).prefetch(tf.data.experimental.AUTOTUNE).batch(N_BATCH).repeat()
    # dataset.shuffle() : 고정 크기의 버퍼를 유지하면서 해당 버퍼에서 다음 요소를 무작위로 선택, 결과 확인을 위해 데이터에 인덱스를 추가

val_dataset = tf.data.TFRecordDataset(val_tfr_dir)
val_dataset = val_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(N_BATCH).repeat()
    # dataset.batch() : batch 사이즈를 ()로 설정. batch가 없으면 리스트에 담기지 않고 데이터를 하나씩 불러옴.
        # example) batch(8) 이면 숫자가 8개씩 들어가고, 이미지라면 1배치당 8개의 이미지가 들어감.


#모델 생성
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

InceptionresnetV2 = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # InceptionResNetV2() : ImageNet 데이터베이스의 백만 개가 넘는 영상에 대한 훈련이 된 컨벌루션 신경망
#mobilenetv2.trainable = False


def create_model():
    model = tf.keras.Sequential()
        # keras.Sequential() : 레이어에 스택을 순서대로 만들어서 적용하는 모델
    model.add(tf.keras.layers.Input([IMG_SIZE, IMG_SIZE, 3], dtype = tf.float32))
        # tf.keras.layers.Input() : keras 텐서를 인스턴스화하는 데에 사용됨
        # dtype : 입력이 문자열(float32)로 예상하는 데이터 형식
    model.add(tf.keras.layers.Lambda(preprocess_input, name='preprocessing', input_shape=(224, 224, 3)))
        # tf.keras.layers.Lambda() : 사용자가 정의한 코드로 layer를 실행할 수 있으며 Sequential model 안에서 임의의 함수로 실행
        # input을 절대값으로 바꿔주는 역할
    model.add(InceptionresnetV2)
        # InceptionresnetV2 : ImageNet 데이터베이스의 백만 개가 넘는 영상에 대한 훈련이 된 컨벌루션 신경망
    model.add(tf.keras.layers.GlobalAveragePooling2D())
        # GlobalAveragePooling2D() : 입력에 대한 전역 평균을 풀링하는 방식
    model.add(tf.keras.layers.Dense(3,activation='softmax'))
        # tf.keras.layers.Dense() : 신경망을 이해할 때 사용하는 모듈. input을 넣었을 때 output으로 바꿔주는 중간다리 역할
    return model

model = create_model()
model.summary()


#하이퍼 파라미터 설정
N_EPOCHS = 10
learning_rate = 0.0001
steps_per_epoch = N_TRAIN/N_BATCH
validation_step = int(np.ceil(N_VAL / N_BATCH))


#오차 함수
def loss_fn(y_true,y_pred):
    return keras.losses.SparseCategoricalCrossentropy()(y_true,y_pred)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=steps_per_epoch*2, decay_rate=0.5, staircase=True)

model.compile(tf.keras.optimizers.RMSprop(lr_schedule),loss=loss_fn, metrics = ['accuracy'])


#모델 학습
start = time.time()
hist = model.fit(train_dataset, steps_per_epoch=steps_per_epoch,
         epochs=N_EPOCHS,
         validation_data=val_dataset,
         validation_steps=validation_step)
end = time.time()


# 모델 테스트 데이터 불러오기
test_dir = os.path.join(camera_dir, 'test')
test0_dir = os.path.join(test_dir, '0')
test1_dir = os.path.join(test_dir, '1')
test2_dir = os.path.join(test_dir, '2')
test_normal_img_files = os.listdir(test0_dir)
test_fire_img_files = os.listdir(test1_dir)
test_smoke_img_files = os.listdir(test2_dir)
test_x = []
test_y = []

for file in test_normal_img_files:
    img_path = os.path.join(test0_dir, file)
    image = Image.open(img_path)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = tf.cast(np.array(image), tf.float32) / 255.
    test_x.append(image)
    test_y.append(0)

for file in test_fire_img_files:
    img_path = os.path.join(test1_dir, file)
    image = Image.open(img_path)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = tf.cast(np.array(image), tf.float32) / 255.
    test_x.append(image)
    test_y.append(1)

for file in test_smoke_img_files:
    img_path = os.path.join(test2_dir, file)
    image = Image.open(img_path)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = tf.cast(np.array(image), tf.float32) / 255.
    test_x.append(image)
    test_y.append(2)

test_x = np.array(test_x)
test_y = np.array(test_y)


#모델 평가
loss_and_metrics = model.evaluate(test_x, test_y, batch_size=N_BATCH)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)


#평가 결과 시각화
#fig, loss_ax = plt.subplots()
#acc_ax = loss_ax.twinx()

#loss_ax.plot(hist.history['loss'], 'y', label='train loss')
#loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
#loss_ax.set_xlabel('epoch')
#loss_ax.set_ylabel('loss')
#loss_ax.legend(loc='upper left')

filenamekey = int(time.time())

plt.plot(hist.history['accuracy'], 'b', label='train accuracy')
plt.plot(hist.history['val_accuracy'], 'g', label='val accuracy')
plt.ylabel('accuracy')
plt.legend(loc='upper left')
plt.text(N_EPOCHS-1,loss_and_metrics[1], 'ta : {}'.format(format(loss_and_metrics[1],".2f")), fontsize=11, bbox=dict(boxstyle='square', color='lightgray'),horizontalalignment='right')
plt.savefig('./results/graph/figure{}_{}.png'.format(data.shape[0],filenamekey))


#모델 저장
model.save('./model/Classification_Model{}'.format(filenamekey))
model.save_weights('./model/Classification_Model_weights{}.h5'.format(filenamekey))
with open('./results/modelsummary{}.txt'.format(filenamekey), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))


#학습 결과 저장
data = data.append({'key':filenamekey,
                    'common train data len':N_N_TRAIN,
                    "fire train data len":N_F_TRAIN,
                    "smoke train data len":N_S_TRAIN,
                    "common validation data len":N_N_VAL,
                    "fire validation data len":N_F_VAL,
                    "smoke validation data len":N_S_VAL,
                    "EPOCHS":N_EPOCHS,
                    "learning_rate_schedule":"ExponentialDecay(initial_learning_rate={},decay_steps={},decay_rate=0.5,staircase=True)".format(learning_rate,steps_per_epoch*2),
                    "batch size":N_BATCH,
                    "loss_func":'SparseCategoricalCrossentropy',
                    "optimizer":"RMSprop",
                    "comment": "inception_resnet_v2",
                    "test_accuracy":loss_and_metrics[1],
                    "train time":format(end-start, ".2f")},ignore_index=True)

data.to_csv('./results/HyperParameter.csv',encoding='euc-kr')
