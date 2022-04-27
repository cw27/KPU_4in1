import socket
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# 모델 불러오기
def loss_fn(y_true, y_pred):
    return keras.losses.MeanSquaredError()(y_true, y_pred)

def local_loss_fn(y_true, y_pred):
    return keras.losses.MeanSquaredError()(y_true, y_pred)

def class_loss_fn(y_true, y_pred):
    return keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)


# Model 클래스를 서브 클래싱하여 __init__에 레이어를 정의하고, 전달 call 구현한다.
# __init__ 메서드에서 층을 만들어 클래스 객체의 속성으로 지정
# call 메서드에 정방향 패스를 정의, __init__ 메서드에서 정의한 층을 사용
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.classification = tf.keras.models.load_model("./model/Classification_Model",
                                                         custom_objects={'loss_fn': local_loss_fn})
        self.classification.load_weights('./model/Classification_Model_weights.h5')
        self.localization = tf.keras.models.load_model("./model/Localization_Model",
                                                       custom_objects={'loss_fn': class_loss_fn})
        self.localization.load_weights('./model/Localization_Model_weights.h5')
        self.con = tf.keras.layers.Concatenate(axis=-1)

    def call(self, x, training=False, mask=None):
        a = self.classification(x)
        b = self.localization(x)

        return self.con([a, b])

model = MyModel()

# TCP 소켓으로부터 메시지 수신
def recvall(sock, count):
    buf = b''

    while count:
        newbuf = sock.recv(count)       # 클라이언트로부터 데이터 수신
        if not newbuf: return '0'

        buf += newbuf
        count -= len(newbuf)
    return buf

IMG_SIZE = 224

# 서버의 IP 및 포트번호
HOST = '192.168.219.102'
PORT = 9999

# TCP 소켓 연결
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 소켓 객체 생성. 주소 체계로 IPv4, 소켓 타입으로 TCP 사용
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 포트가 사용 중이라 연결할 수 없다는 WinError 10048 에러 해결을 위해 필요
server.bind((HOST, PORT))
    # bind 함수는 소켓을 특정 네트워크 인터페이스와 포트 번호에 연결하는 데에 사용됨
    # HOST는 hostname, ip 주소, 빈 문자열이 될 수 있고, 빈 문자열일 때는 모든 네트워크 인터페이스로부터의 접속을 허용함
    # PORT는 1부터 65535 사이의 숫자 사용 가능
server.listen()
    # 서버가 클라이언트의 접속을 허용하도록 함
client, addr = server.accept()
    # accept 함수에서 대기하다가 클라이언트가 접속하면 새로운 소켓을 리턴

print('Connected by', addr) # 접속한 클라이언트의 주소

while True:
    # 서버가 클라이언트로 메시지를 보내면 클라이언트가 영상(문자열) 한 장을 보냄
    message = '1'
    client.send(message.encode())

    # 영상(문자열)을 이미지(opencv)로 변환
    length = recvall(client, 16)
    stringData = recvall(client, int(length))
    data = np.frombuffer(stringData, dtype='uint8')
    decimg = cv2.imdecode(data, 1)

    # 이미지(opencv)를 모델에 맞는 데이터로 변환
    image = Image.fromarray(decimg)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = tf.cast(np.array(image), tf.float32) / 255.
    image = image[tf.newaxis, ...]
    decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

    # 데이터 입력 및 분류
    prediction = model.predict(image)
    pred_label = int(tf.argmax(prediction[0][:3]))
    pred_local = prediction[0][3:]

    # 원본 이미지에 분류 결과 정보(화재여부, 화재 지역) 합성
    image = image[0]

    if pred_label == 0:
        decimg = cv2.rectangle(decimg, (0, 0), (150, 40), (255, 255, 255), cv2.FILLED)
        decimg = cv2.putText(decimg, "Non-Fire", (0, 30), cv2.FONT_ITALIC, 1, (0, 0, 255), 3)

    elif pred_label == 1:
        decimg = cv2.rectangle(decimg, (0, 0), (80, 40), (255, 255, 255), cv2.FILLED)
        decimg = cv2.putText(decimg, "Fire!!", (0, 30), cv2.FONT_ITALIC, 1, (0, 0, 255), 3)
        pred_x = pred_local[0]
        pred_y = pred_local[1]
        pred_w = pred_local[2]
        pred_h = pred_local[3]

        height, width, _ = decimg.shape

        pred_xmin = int((pred_x - pred_w / 2.) * width)
        pred_ymin = int((pred_y - pred_h / 2.) * height)
        pred_xmax = int((pred_x + pred_w / 2.) * width)
        pred_ymax = int((pred_y + pred_h / 2.) * height)
        print(pred_local)
        decimg = cv2.rectangle(decimg, (pred_xmin, pred_ymin), (pred_xmax, pred_ymax), (0, 0, 255), 2)

    elif pred_label == 2:
        decimg = cv2.rectangle(decimg, (0, 0), (120, 40), (255, 255, 255), cv2.FILLED)
        decimg = cv2.putText(decimg, "SMOKE!", (0, 30), cv2.FONT_ITALIC, 1, (0, 0, 255), 3)

    # 합성 이미지 출력
    cv2.imshow('video', decimg)

    key = cv2.waitKey(1)

    if key == 27:
        break

client.close()
server.close()
