# NEW FEATURES
# -------------------------------------------------------------------------------
# Name:        Outputting format for orders
# Purpose:     Outputting format for orders training yoloV3
# Author:      aka9
# Created:     28/07/2019

#
# -------------------------------------------------------------------------------

# glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일의 이름을 리스트 형식으로 반환한다.
import glob
# os 모듈을 이용하여 운영체제가 지원하는 기능을 사용하여 디렉토리를 만들거나 옮기기 위한 기능을 제공한다.
import os
# 파이썬 인터프리터가 제공하는 변수나 함수를 제어할 수 있는 방법을 제공한다.
import sys


def process(current_dir):
    # current_dir = './images'

    # Percentage of images to be used for the test set
    percentage_test = 10

    # Create and/or truncate train.txt and test.txt
    file_train = open('../results/train.txt', 'w')
    file_test = open('../results/test.txt', 'w')

    # Populate train.txt and test.txt
    counter = 1
    # 반올림한다.
    index_test = round(100 / percentage_test)
    print("Warning: we assume that all pictures to be processed are jpgs")
    # 실제로 동시에 모든 값을 저장하지 않고 glob과 동일한 값을 생성한다.
    for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
        # 파일의 경로를 os.path.splitext 함수의 매개변수로 넣어두면 두 개의 값을 반환되고 두 번째 값이 파일의 확장자
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test:
            counter = 1
            # 삽입()안의 string을 file_test에 적는다.
            file_test.write(current_dir + "/" + title + '.jpg' + "\n")
        else:
            # ()안의 string을 file_train에 적는다.
            file_train.write(current_dir + "/" + title + '.jpg' + "\n")
            counter = counter + 1


def main():
    # sys.argv는 프로그램을 실행할 때 입력값을 읽어 들일 수 있는 파이썬 라이브러리
    array = sys.argv[1:]
    if len(array) > 1:
        print("Not the right number of arguments, use -h for usage recommendations")
    elif len(array) == 0:
        print("Not the right number of arguments, use -h for usage recommendations")
    elif array[0] == '-h':
        print('simply write the path_to_folder as an argument')
    else:
        # 입력받은 경로가 존재하지 않으면 False
        if not os.path.exists(array[0]):
            print('folder not found')
        # 입력받은 경로가 존재하면 True
        else:
            process(array[0])


if __name__ == '__main__':
    main()
