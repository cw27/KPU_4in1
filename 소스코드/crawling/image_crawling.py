from selenium import webdriver
import chromedriver_autoinstaller
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import quote_plus
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import os

# 크롤링할 이미지 최대 개수 (이미지는 NUM 이하로 크롤링됨)
NUM = 1000000

# 크롬드라이버 버전 확인
chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0]

try:
    driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe')
    
# try 블록 수행 중 오류 발생하면 except 블록 수행해서 크롬드라이버 다운로드
except:
    chromedriver_autoinstaller.install(True)
    driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe')

# 브라우저에서 사용되는 엔진 자체에서 파싱되는 시간을 기다려 주는 메소드 (10초 안에 웹피이지를 load 하면 바로 넘어가거나, 10초 기다림)
driver.implicitly_wait(10)

# path(url)에 있는 데이터 모두 추출
def loadHtml(path):
    # path(url)로 이동
    driver.get(path)
    # 3초 안에 웹페이지를 load 하면 바로 3초 넘어가거나, 3초 기다림
    driver.implicitly_wait(3)
    # 브라우저에 보이는 그대로의 HTML, 크롬 개발자 도구의 Element 탭 내용과 동일
    html = driver.page_source
    # 인터넷 문서의 구조에서 명확한 데이터를 추출하고 처리하는 가장 쉬운 라이브러리
    soup = BeautifulSoup(html, 'html.parser')
    # 해당 태그 모두 추출 (기준에 맞는 태그를 모두 가져오기 때문에 리스트 타입 반환)
    return soup.find_all('img')


def crawlingImage(n, savepath, failStack):
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    img = soup.find_all('img')
    print(len(img), n)
    if len(img) <= n:
        # chrome 창 스크롤을 내려서 이미지가 렌더링 되도록 함
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        # 결과 더보기 버튼이 있으면 클릭
        result = driver.execute_script(
            '''
            if(document.getElementsByClassName("mye4qd")[0].getAttribute("value")==="결과 더보기")
                document.getElementsByClassName("mye4qd")[0].click()
            '''
        )
        failStack += 1
        if failStack > 5:
            return n, -1
        else:
            return n, failStack

    for i in img[n:]:
        print(n)
        try:
            imgUrl = i['src']
        except KeyError:
            try:
                imgUrl = i['data-src']
            except KeyError:
                continue
        # url에서 이미지를 읽어서 지정된 경로에 저장
        # with urlopen(파일 경로) as 파일 객체:
        with urlopen(imgUrl) as f:
            with open(savepath + str(n) + '.jpg', 'wb') as h:  # w - write b - binary
                # 파일 스트림으로부터 해당 위치의 모든 문자를 읽어오는 함수
                img = f.read()
                # 매개변수로 파일에 넣은 문자열 받음
                h.write(img)
        # 저장된 이미지 출력
        # with Image.open(savepath + str(n)+'.jpg') as image:
        #    plt.imshow(image)
        #    plt.show()
        n += 1
        # driver.execute_script("window.scrollTo(0, {})".format(n*(231//4)))

        if n > NUM:
            break
    failStack = 0
    return n, failStack

img = loadHtml('https://www.google.com/search?q=fire+outbreak+kitchen&tbm=isch&ved=2ahUKEwjDjevMntzwAhW0yosBHZE0Cs4Q2-cCegQIABAA&oq=fire+outbreak+kitchen&gs_lcp=CgNpbWcQA1CB7ARYgewEYInuBGgAcAB4AIABiAGIAYgBkgEDMC4xmAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=-muoYMOoDbSVr7wPkemo8Aw&bih=754&biw=1536&rlz=1C1OKWM_koKR870KR870&hl=ko')
n = 0
failStack = 0
while n < NUM:
    n, failStack = crawlingImage(n, './data/crawled_images/1/img2', failStack)
    if failStack == -1:
        print("{}개의 이미지 크롤링 성공".format(n))
        break


# RGB 제외한 이미지 삭제
# 현재 커서가 위치해 있는 디렉터리
cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data')
crawled_dir = os.path.join(data_dir, 'crawled_images')
class0_dir = os.path.join(crawled_dir, '0')     # 관련 없는 이미지
class1_dir = os.path.join(crawled_dir, '1')     # 불꽃 이미지 (전체 - 관련 없는 이미지 - 연기 이미지)
class2_dir = os.path.join(crawled_dir, '2')     # 연기 이미지

normal_img_files = os.listdir(class0_dir)       # 관련 없는 이미지 파일
fire_img_files = os.listdir(class1_dir)         # 불꽃 이미지
smoke_img_files = os.listdir(class2_dir)        # 연기 이미지

# 관련 없는 이미지에서 RGB 없는 흑백 이미지 삭제
nn = 0
for img in normal_img_files:
    img_path = os.path.join(class0_dir, img)
    image = Image.open(img_path)
    image_mode = image.mode
    # RGB 없는 이미지 삭제
    if image_mode != 'RGB':
        print(img, image_mode)
        # ndarray의 데이터 형태가 다를 경우에만 복사
        image = np.asarray(image)
        print(image.shape)
        os.remove(img_path)
        nn += 1
print("일반 이미지 {}개 삭제".format(nn))

# 화재 이미지에서 RGB 없는 흑백 이미지 삭제
fn = 0
for img in fire_img_files:
    img_path = os.path.join(class1_dir, img)
    image = Image.open(img_path)
    image_mode = image.mode
    if image_mode != 'RGB':
        print(img, image_mode)
        image = np.asarray(image)
        print(image.shape)
        os.remove(img_path)
        fn += 1
print("화재 이미지 {}개 삭제".format(fn))

# 연기 이미지에서 RGB 없는 흑백 이미지 삭제
sn = 0
for img in smoke_img_files:
    img_path = os.path.join(class2_dir,img)
    image = Image.open(img_path)
    image_mode = image.mode
    if image_mode != 'RGB':
        print(img, image_mode)
        image = np.asarray(image)
        print(image.shape)
        os.remove(img_path)
        sn += 1
print("연기 이미지 {}개 삭제".format(sn))