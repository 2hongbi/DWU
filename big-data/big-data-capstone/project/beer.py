from multiprocessing import Manager

import pandas as pd
import numpy as np
import time

import re

from selenium import webdriver
from selenium.webdriver.common.by import By

from multiprocessing import Pool
from selenium.webdriver.support import expected_conditions as EC

import chromedriver_autoinstaller

import parmap

# Kloud 추가
beer_list = [
 'Kirin ichiban', 'Sapporo Premium Beer / Draft Beer', 'stella artois', 'guinness braught',
 'cass fresh',
 'stout', 'dry finish', 'max hite', 'hite extra cold',
  'TIGER REDLER',
 'PEEPER B IPA',
 'TSINGTAO WHEAT BEER', 'Carlsberg',
'Peroni Nastro Azzurro',
 'Guinness original',
 'Filite', 'SEOULITE ALE',
 'Bali Hai Premium Larger', 'Apostel Brau',
 'Egger Zwickl', 'Egger Marzenbier', 'Holsten Premium Beer', 'Franzisaner Hefe-Weissbier',
 'Egger Radler Grapefruit', 'Barvaria Premium', 'Barvaria 8.6',
  'Leffe Brown',
 'Platinum Pale Ale', 'Ambar Especial Larger',
 'Schöfferhofer Grapefruit',
 'BURGE MEESTER',
'Jeju Coffee Golden Ale',  'Leffe Blonde / Blond', 'Hop House 13 Lager', 'Happoshu Filgood', 'Cafri', 'Blue Girl',  'ARK Cosmic Dancer', 'ARK Seoulite Ale', 'Kabrew Golden Ale',
'Schneider Weisse Tap 05 - Hopfenweisse Weizendoppelbock',
'Paulaner Hefe-Weissbier',
'Goose Island Duck Duck Goose',
'Desperados',
'Desperados Red',
'Desperados Mojito',
'Desperados Lime',
 'ARK Be High IPA', 'ARK Hug Me', 'ARK Black Swan',
 'Cass Light', 'Kabrew Gyeongbokgung Royal Pride IPA',
 'Kabrew Namsan Mountain Premium Citra Ale', 'Kabrew Kumiho Peach Ale']

beer_list = ['guinness draught']
#  'guinness original', 'hoegarden', 'pilsner urquell', 'stella artois',
#             'weissbier naturtrub'


# beer_list = ['kloud']

# 데이터프레임으로 저장
beer_list = pd.DataFrame(data=beer_list, columns=['검색이름'])

# 크롤링 할 경로 설정
url = 'https://www.ratebeer.com/search?tab=beer'


def crawl(k):
    # 데이터 프레임 생성
    data = pd.DataFrame(data=[], columns=['맥주정보', '검색이름', '맥주이름'])

    # url open
    beer = beer_list['검색이름'].iloc[k]
    print('url_open... {0} 맥주 데이터를 수집합니다..'.format(beer))
    driver = webdriver.Chrome()
    driver.get(url)
    driver.set_window_size(900, 900)

    # 1번 사진에 해당 : 맥주 검색
    time.sleep(2)
    element = driver.find_element(By.XPATH, '//*[@id="root"]/div[2]/header/div[2]/div[1]/div[2]/div/div/input')
    time.sleep(2)
    element.click()
    time.sleep(2)
    element.send_keys(beer)
    time.sleep(3)

    driver.implicitly_wait(10)
    # 2번 사진에 해당 : 상품 선택
    driver.find_element(By.XPATH, '//*[@id="root"]/div[2]/header/div[2]/div[1]/div[2]/div/div[2]/a[1]/div/div[2]').click()

    # 3번 사진에 해당 : 상품 이름 수집
    time.sleep(3)
    beer_name = driver.find_element(By.CSS_SELECTOR, '.MuiTypography-root.Text___StyledTypographyTypeless-bukSfn.pzIrn.text-500.colorized__WrappedComponent-hrwcZr.hwjOn.mt-3.MuiTypography-h4').text

    print(beer_name)



    # 맥주 회사
    time.sleep(2)
    company = driver.find_element(By.CSS_SELECTOR, '.MuiTypography-root.Text___StyledTypographyTypeless-bukSfn.pzIrn.colorized__WrappedComponent-hrwcZr.liJcHu.Anchor___StyledText-uWnSM.eseQug.MuiTypography-subtitle1').text
    print(company)

    # 원산지
    time.sleep(2)
    origin = driver.find_element(By.XPATH, '//*[@id="root"]/div[2]/div[2]/div/div/div/div[2]/div[1]/div/div[2]/div[1]/div/div[2]/div[3]/div[1]').text
    print(origin)

    # 맥주 종류
    time.sleep(2)
    beer_type = driver.find_element(By.CSS_SELECTOR, '.MuiTypography-root.Text___StyledTypographyTypeless-bukSfn.kbrPIo.colorized__WrappedComponent-hrwcZr.liJcHu.Anchor___StyledText-uWnSM.eseQug.MuiTypography-caption').text
    print(beer_type)

    # 도수
    time.sleep(2)
    dosu = driver.find_element(By.XPATH, '//*[@id="root"]/div[2]/div[2]/div/div/div/div[2]/div[1]/div/div[2]/div[1]/div/div[2]/div[3]/div[2]/div[2]').text
    print(dosu)


    # ibu
    time.sleep(2)
    ibu = ''
    try:
        ibu = driver.find_element(By.XPATH, '//*[@id="root"]/div[2]/div[2]/div/div/div/div[2]/div[1]/div/div[2]/div[1]/div/div[2]/div[3]/div[2]/div[4]').text
        print(ibu)
    except Exception:
        pass

    # 평점
    time.sleep(2)
    rating = driver.find_element(By.CSS_SELECTOR, '.MuiTypography-root.Text___StyledTypographyTypeless-bukSfn.pzIrn.text-500.colorized__WrappedComponent-hrwcZr.hwjOn.mr-2.MuiTypography-body2').text
    print(rating)

    # 설명
    time.sleep(2)
    desc = driver.find_element(By.CSS_SELECTOR, '.MuiTypography-root.Text___StyledTypographyTypeless-bukSfn.pzIrn.colorized__WrappedComponent-hrwcZr.hwjOn.pre-wrap.MuiTypography-body2').text
    print(desc)

    # show more click
    driver.find_element(By.XPATH, '//*[@id="root"]/div[2]/div[2]/div/div/div/div[2]/div[1]/div/div[4]/button/span[1]').click()

    # 칼로리
    time.sleep(3)
    kcal = ''
    try:
        kcal = driver.find_element(By.XPATH, '//*[@id="root"]/div[2]/div[2]/div/div/div/div[2]/div[1]/div/div[3]/div/div/div/div/div[1]/div[2]/div[2]').text
        print(kcal)
    except Exception:
        pass

    error_cnt = 0

    while 1:
        try:
            # 4번 사진에 해당 : 전체 리뷰 개수 수집
            time.sleep(3)

            string = driver.find_element(By.CSS_SELECTOR, '.MuiTypography-root.Text___StyledTypographyTypeless-bukSfn.pzIrn.text-500.colorized__WrappedComponent-hrwcZr.hwjOn.MuiTypography-h6').text
            print(string)

            # ,가 포함되어 있는지에 대한 로직
            extract = re.compile('[0-9]*,*[0-9]+')
            str_num = extract.findall(string)
            str_num = str_num[0]
            print(str_num)

            print('성공... while문을 탈출합니다.')
            break
        except :
            print('오류 발생.. 다시 시작합니다.')

            error_cnt += 1

            if error_cnt == 5:
                print('연속된 오류로 다음 맥주로 넘어갑니다...')
                return

    if ',' in str_num:
        str_num = str_num.split(',')
        str_num = int(str_num[0]+str_num[1])
        num = str_num
    else:
        num = int(str_num)

    # 5번 사진에 해당 : Score breakdown 클릭
    time.sleep(3)
    element = driver.find_element(By.XPATH, '//*[@id="root"]/div[2]/div[2]/div/div/div/div[2]/div[4]/div/div[2]/div[1]/div[2]')
    time.sleep(3)
    # 해당 element로 이동하는 코드입니다. 반드시 적어주세요.
    driver.execute_script("arguments[0].click();", element)

    # 수집할 Page 수를 계산합니다.
    page_num = num // 15 + 1


    for i in range(page_num):
        print(i+1, '번째 페이지입니다.')

        # 6번 사진에 해당 : 전체 맥주 정보를 통째로 수집
        time.sleep(3)
        beer_info = driver.find_elements(By.CSS_SELECTOR,'.px-4.fj-s.f-wrap')

        tmp = []

        # 수집한 것을 데이터프레임에 저장
        for i in range(len(beer_info)):
            tmp.append(beer_info[i].text)

        tmp = pd.DataFrame(data=tmp, columns=['맥주정보'])
        tmp['맥주이름'] = beer_name
        tmp['검색이름'] = beer
        tmp['맥주회사'] = company
        tmp['맥주원산지'] = origin
        tmp['맥주종류'] = beer_type
        tmp['맥주도수'] = dosu
        tmp['맥주IBU'] = ibu
        tmp['맥주평점'] = rating
        tmp['맥주설명'] = desc
        tmp['맥주칼로리'] = kcal

        data = pd.concat([data, tmp])

        # 다음 페이지로 넘어가기 : 7번 사진에 해당합니다.
        # div, span, title 태그 후 속성은 class 외에도 사용 가능
        try :
            element = driver.find_element(By.XPATH, '//button[@title="Next page"]/span[@class="MuiIconButton-label"]')
            time.sleep(3)
            driver.execute_script("arguments[0].click();", element)
        except:
            print('마지막 페이지입니다.')

    # 데이터가 중복 수집될 경우 리뷰 수 만큼만 Cut
    if num != len(data):
        data = data[:num]

    print('리뷰수 : ', num, '수집된 리뷰수 : ', len(data))

    # 데이터를 csv, excel 파일로 저장합니다.
    result = pd.merge(data, beer_list, on='검색이름', how='left')
    result.to_csv("./data/beer_n_"+str(beer_name)+".csv", encoding='utf-8')
    result.to_excel("./data/beer_n_"+str(beer_name)+".xlsx")

    driver.quit()
    print(beer_name + '종료')
    return result


if __name__ == '__main__':
    # 크롬 드라이버 install
    chromedriver_autoinstaller.install()

    num_cores = 4

    input_list = range(len(beer_list))

    # parmap.map(crawl, input_list, pm_pbar=True, pm_processes=num_cores)
    # 맥주 리스트 개수만큼 데이터 수집
    for k in range(len(beer_list)):
        result = crawl(k)
