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


NUM = 100 #크롤링할 이미지 최대 개수(이미지는 NUM이하로 크롤링됨)


chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0]

try:
    driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe')
except:
    chromedriver_autoinstaller.install(True)
    driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe')

driver.implicitly_wait(10)
#driver = webdriver.Chrome("C:/python/chromedriver.exe")
#driver.implicitly_wait(3)

def loadHtml(path):
    driver.get(path)
    driver.implicitly_wait(3)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
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
        with urlopen(imgUrl) as f:
            with open(savepath + str(n) + '.jpg', 'wb') as h:  # w - write b - binary
                img = f.read()
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
    n, failStack = crawlingImage(n,'./images/1/img2',failStack)
    if failStack==-1:
        print("{}개의 이미지 크롤링 성공".format(n))
        break


#RGB를 제외한 이미지 삭제
cur_dir = os.getcwd()
#data_dir = os.path.join(cur_dir, 'tutorial')
crawled_dir = os.path.join(cur_dir, 'images')
class0_dir = os.path.join(crawled_dir,'0')
class1_dir = os.path.join(crawled_dir,'1')
class2_dir = os.path.join(crawled_dir,'2')

normal_img_files = os.listdir(class0_dir)
fire_img_files = os.listdir(class1_dir)
smoke_img_files = os.listdir(class2_dir)

nn=0
for img in normal_img_files:
    img_path = os.path.join(class0_dir,img)
    image = Image.open(img_path)
    image_mode = image.mode
    if image_mode != 'RGB':
        print(img, image_mode)
        image = np.asarray(image)
        print(image.shape)
        os.remove(img_path)
        nn+=1
print("일반 이미지 {}개 삭제".format(nn))

fn = 0
for img in fire_img_files:
    img_path = os.path.join(class1_dir,img)
    image = Image.open(img_path)
    image_mode = image.mode
    if image_mode != 'RGB':
        print(img, image_mode)
        image = np.asarray(image)
        print(image.shape)
        os.remove(img_path)
        fn+=1
print("화재 이미지 {}개 삭제".format(fn))

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
        sn+=1
print("연기 이미지 {}개 삭제".format(sn))
