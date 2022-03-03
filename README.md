### 딥러닝을 활용한 화재 감지 시스템 🔥  
- - -
##### 본 프로젝트는 딥러닝을 활용하여 영상 속 화재를 감지하는 시스템이다. 카메라 영상 데이터를 서버로 전송한 후 미리 학습되어있는 신경망을 이용하여 카메라 영상으로만 분석한 후에 화재 여부를 판단하고 웹으로 화재 경보 알림과 화재 위치 및 화재 크기에 대한 정보를 제공한다.  

#### 개발 목표  
- - -
###### 따라서 본 프로젝트에서는 기존의 사회적 생산 기반을 활용할 수 있되, 영상 및 감지 센서를 복합적으로 이용해 높은 수준의 신뢰도를 보이는 화재 감지 시스템을 구현하는 것을 목표로 한다.  

#### 시나리오  
- - -
###### 0. CCTV 동작  
###### 1. 화재 발생  
######  1.1 학습된 알고리즘 분석(불꽃 또는 연기)을 통해 화재 상황 1차 인식  
######  1.2 부착된 불꽃 감지 센서와 가스 감지 센서로 화재 상황 2차 인식  
###### 2. 화재 발생 확인  
######  2.1 화재 발생 전, 화재 발생 후 상황 녹화  
######  2.2 화재 경보  
######  2.3 사용자 대상 화재 알림 및 각 IP 주소를 토대로 한 화재 위치와 정보 전송  

#### 개발 환경  
- - -
###### OS  
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)  
###### Language  
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  
###### DB  
![MySQL](https://img.shields.io/badge/mysql-%2300f.svg?style=for-the-badge&logo=mysql&logoColor=white)  
###### IDE  
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)  
