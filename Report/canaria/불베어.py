# -*- coding: utf-8 -*-
"""
Created on Sun May 15 19:55:11 2022

@author: 82106
"""
#%%라이브러리 소환
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%데이터 소환
ism = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\상관관계_파이널.xlsx',index_col=0 , sheet_name=0)
유가 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\상관관계_파이널.xlsx',index_col=0, sheet_name=5)
환율 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\상관관계_파이널.xlsx',index_col=0, sheet_name=6)
금리차 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\상관관계_파이널.xlsx',index_col=0, sheet_name=4)
high_yield = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\상관관계_파이널.xlsx',index_col=0, sheet_name=1)
빅스 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\상관관계_파이널.xlsx',index_col=0, sheet_name=2)
경상수지 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\상관관계_파이널.xlsx',index_col=0, sheet_name=3)

ism_c = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\표준화_파이널.xlsx',index_col=0 , sheet_name=0)
유가_c = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\표준화_파이널.xlsx',index_col=0, sheet_name=5)
환율_c = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\표준화_파이널.xlsx',index_col=0, sheet_name=6)
금리차_c = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\표준화_파이널.xlsx',index_col=0, sheet_name=4)
high_yield_c = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\표준화_파이널.xlsx',index_col=0, sheet_name=1)
빅스_c = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\표준화_파이널.xlsx',index_col=0, sheet_name=2)
경상수지_c = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\표준화_파이널.xlsx',index_col=0, sheet_name=3)

코스피수익률 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\코스피_최종.xlsx',index_col=0)
indexing = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\index.xlsx',index_col=0)


dumdum = [ism,유가, 환율, 금리차, high_yield, 빅스, 경상수지]





#%% bull/bear


cor=(유가+환율+빅스+high_yield+금리차)/5
dis=(유가_c+환율_c+빅스_c+high_yield_c+금리차_c)/5
distance = dis+1-cor

#train / test 구분
testset = 코스피수익률.iloc[:108,: ]
trainset = 코스피수익률.iloc[108:,: ]

#train/test 각각 하락/횡보/상승장 판독 -> iqr로 일단 기준설정
plt.hist(코스피수익률['코스피'], bins = 20, cumulative=False)
plt.show()
코스피수익률['코스피'].describe()
코스피수익률['코스피'].skew()

bear =코스피수익률[코스피수익률['코스피'] <= -0.024153].index
bull = 코스피수익률[코스피수익률['코스피'] >  0.070720].index


#인덱스 리스트
index_list=distance.index


#시장종류, dateindex를 넣어서 해당 시점과 비슷한 시점들 색출-> 
#그 지점들이 동일한 분류(에 들어가 있는지 확인해주는 함수

def eval(market, dateindex) :
    index_num=indexing.index[indexing['date'] == dateindex][0]
    for i in market:  
        거리순위 = pd.DataFrame([index_list, distance.iloc[index_num]], 
                            index = ['날짜', '거리']).T
        거리순위 = 거리순위.set_index('날짜')
        국면인식_날짜 = 거리순위.sort_values(by='거리').iloc[1:5]
    acc = 0 
    for j in range(0,1):        
         acc += int(국면인식_날짜.index[j] in market)   
    return acc/1
    


bear_acc = 0
for i in range(54):
    bear_acc += eval(bear, bear[i])

bear_acc = bear_acc/54
bear_acc

bull_acc = 0
for i in range(54):
    bull_acc += eval(bull, bull[i]) 

bull_acc = bull_acc/54
bull_acc
