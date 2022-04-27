import socket
import cv2
import numpy
from queue import Queue
from _thread import *

IMG_SIZE = 224
enclosure_queue = Queue()

#카메라로부터 영상을 읽어서 queue에 넣는 쓰레드가 수행하는 함수
def webcam_input(queue):
    capture = cv2.VideoCapture('./fire and smoke.mp4')
    speed = 6
    i = 0
    while True:
        ret, frame = capture.read()

        if ret == False:
            continue
        i += 1

        if (i % speed) != 0:
            continue

        frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', frame2, encode_param)

        # 영상을 문자열 자료형으로 변환 후 저장
        data = numpy.array(imgencode)
        stringData = data.tostring()
        queue.put(stringData)

        # 영상 출력
        cv2.imshow('image', frame)
        key = cv2.waitKey(1)

        if key == 27:
            return

# queue에서 영상을 읽어서 client_socket을 이용해 서버로 전송하는 쓰레드가 수행하는 함수
def webcam_transmit(client_socket, queue):
    while True:
        try:
            # 서버로부터 메시지를 받을 때마다 영상을 한 장씩 전송
            data = client_socket.recv(1024)     # 서버로부터 데이터를 수신
            if not data:
                print('Disconnected')
                break

            # 영상(문자열) 전송
            stringData = queue.get()
            client_socket.send(str(len(stringData)).ljust(16).encode())     # 서버로 데이터 송신
            client_socket.send(stringData)      # 서버로 데이터 송신

        except ConnectionResetError as e:
            print('Disconnected')
            break

    client_socket.close()   # client_socket 연결 종료

# 서버의 IP 주소 및 포트번호
HOST = '192.168.160.237'
PORT = 9999

# 소켓 생성 및 연결
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 소켓 객체 생성. 주소 체계로 IPv4, 소켓 타입으로 TCP 사용
client_socket.connect((HOST, PORT))
    # 지정한 HOST와 PORT를 사용하여 서버에 접속

print('client start')

# 영상 읽기 쓰레드 생성
i = start_new_thread(webcam_input, (enclosure_queue,))
# 영상 전송 쓰레드 생성
t = start_new_thread(webcam_transmit, (client_socket, enclosure_queue,))