from PIL import Image
import imagehash
import os
import numpy as np

# 파이썬 이미지 처리 PIL(pillow)를 이용한다.
# imagehash를 이용하여 비슷한 이미지를 찾는다.
# - Image hash 알고리즘은 이미지 구조의 휘도, 색의 분배와 비율을 분석한다.
# os 모듈을 이용해 운영체제에서 제공되는 여러 기능을 파이썬 코드 내에서 수행할 수 있도록 한다.
# numpy를 이용하면 배열과 행렬의 사칙연산을 사용할 수 있다.


class DuplicateRemover:
    # 폴더 이름과 hash 크기를 초기화한다.
    def __init__(self, dirname, hash_size=8):
        self.dirname = dirname
        self.hash_size = hash_size

    # 지정한 디렉토리 내 모든 파일과 디렉토리 리스트(list)를 리턴한다.
    def find_duplicates(self):
        """
        Find and Delete Duplicates
        """

        fnames = os.listdir(self.dirname)
        hashes = {}
        duplicates = []
        print("Finding Duplicates Now!\n")
        # fnames 리스트 안에 있는 이미지를 반복문을 이용하여 관리한다.
        for image in fnames:
            # hash_size를 회색 이미지로 변화, 축소하고 평균값을 계산
            # 왼쪽에서 오른쪽으로 픽셀을 하나씩 검사한다.
            with Image.open(os.path.join(self.dirname, image)) as img:
                temp_hash = imagehash.average_hash(img, self.hash_size)
                if temp_hash in hashes:
                    print("Duplicate {} \nfound for Image {}!\n".format(image, hashes[temp_hash]))
                    duplicates.append(image)
                # 중복이 아니면 hashes[temp_hash]에 이미지를 넣는다.
                else:
                    hashes[temp_hash] = image

        if len(duplicates) != 0:
            a = input("Do you want to delete these {} Images? Press Y or N:  ".format(len(duplicates)))
            space_saved = 0
            if (a.strip().lower() == "y"):
                for duplicate in duplicates:
                    # 파일의 크기(size)를 받아온다.
                    space_saved += os.path.getsize(os.path.join(self.dirname, duplicate))

                    # os 라이브러리를 사용하여 중복을 제거한다.
                    os.remove(os.path.join(self.dirname, duplicate))
                    print("{} Deleted Succesfully!".format(duplicate))

                print("\n\nYou saved {} mb of Space!".format(round(space_saved / 1000000), 2))
            else:
                print("Thank you for Using Duplicate Remover")
        else:
            print("No Duplicates Found :(")

# 현재 파일에 있는 images 안에 있는 파일 /1을 참조한다.
dirname = "./images/1"

# Remove Duplicates 중복 제거
# dr 변수 안에 dirname을 초기화한다.
dr = DuplicateRemover(dirname)
# 중복을 찾고 제거한다.
dr.find_duplicates()