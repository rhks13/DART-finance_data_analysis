# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:25:43 2022

@author: 82106
"""

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

코스피 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\코스피_최종.xlsx',index_col=0)
indexing = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\index.xlsx',index_col=0)
코스피1 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\코스피_1.xlsx',index_col=0)
코스피_종가 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\코스피_종가.xlsx',index_col=0)







#%% bull/bear


cor=(유가+환율+금리차+high_yield+경상수지)/5
dis=(유가_c+환율_c+금리차_c+high_yield_c+경상수지_c)/5

distance = dis+1-cor

#train / test 구분
testset = 코스피.iloc[:108,: ]
trainset = 코스피.iloc[108:,: ]
plt.hist(testset, bins = 20, cumulative=False)
plt.hist(trainset, bins = 20, cumulative=False)
test_bear = testset[testset['코스피'] <= -0.037844].index
test_bull = testset[testset['코스피'] > 0.051298].index
train_bear = trainset[trainset['코스피'] <= -0.005352].index
train_bull = trainset[trainset['코스피'] > 0.098204].index


#후에 쓸일이 있습니다. 224개의 인덱스
index_list=distance.index

#%% 상관관계

indexing_train = indexing[108:]
indexing_test = indexing[:108]
model_test = distance[:108]
model_train = distance[108:]
list_train = []
list_test = []

def bearcorr(dateindex) :
    #파라미터로 입력된 데이터인덱스의 수익률 추가
    list_test.append(코스피['코스피'][dateindex])
    #입력된 인덱스에 맞는 테스트인덱스의 넘버 구하기
    index_num=indexing_test.index[indexing_test['date'] == dateindex][0]
    
    거리순위 = pd.DataFrame([index_list[108:], model_train.iloc[:,index_num]], index = ['날짜', '거리']).T
    거리순위 = 거리순위.set_index('날짜')
    국면인식_날짜 = 거리순위.sort_values(by='거리').iloc[1:13]
    
    acc = 0 
    for i in range(0,6):        
         acc+=코스피['코스피'][국면인식_날짜.index[i]]  
    list_train.append(acc/6)

def bullcorr(dateindex) :
    #파라미터로 입력된 데이터인덱스의 수익률 추가
    list_test.append(코스피['코스피'][dateindex])
    #입력된 인덱스에 맞는 테스트인덱스의 넘버 구하기
    index_num=indexing_test.index[indexing_test['date'] == dateindex][0]
    
    거리순위 = pd.DataFrame([index_list[:108], model_train.iloc[:,index_num]], index = ['날짜', '거리']).T
    거리순위 = 거리순위.set_index('날짜')
    국면인식_날짜 = 거리순위.sort_values(by='거리').iloc[1:13]
    
    acc = 0 
    for i in range(0,6):        
         acc+=코스피['코스피'][국면인식_날짜.index[i]]  
    list_train.append(acc/6)


for i in range(27):
    bearcorr(model_test.index[i])
for i in range(27):
    bullcorr(model_test.index[i])   
np.corrcoef(list_train, list_test)[0,1]


plt.plot(list_test,list_train,"or")
plt.ylabel('list_train')
plt.xlabel('list_test')
plt.title('correlation')
print(np.corrcoef(list_train, list_test)[0,1])

#%% 히트맵
distance.to_excel("distance.xlsx")
distance_1 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\distance.xlsx',index_col=0)

sns.heatmap(distance_1, cmap="RdYlGn_r")
plt.title('Macro Face Recognition', fontsize=50)
#%% 결론

import seaborn as sns
plt.style.use(['default'])
sns.set(rc = {'figure.figsize':(50,28)})
from datetime import datetime
from pandas_datareader import data
plt.plot(코스피_종가['코스피'])
range_list = [('2008-01-31 00:00:00','2008-02-28 00:00:00'), ('2021-11-30 00:00:00','2021-12-30 00:00:00'), ('2008-02-29 00:00:00','2008-03-31 00:00:00'), ('2008-03-31 00:00:00','2008-04-30 00:00:00'), ('2019-07-31 00:00:00','2019-08-31 00:00:00'), ('2019-05-31 00:00:00','2019-06-30 00:00:00')]


for (start, end) in range_list:
    plt.axvspan(start, end, color='gray', alpha=0.5)
#%% 쌉결론
i_list=['2008-01-31 00:00:00', '2021-11-30 00:00:00', '2008-02-29 00:00:00', '2008-03-31 00:00:00', '2019-07-31 00:00:00', '2019-05-31 00:00:00']
income=0;

for i in i_list:
    income += 코스피.loc[i]['코스피'][0]
    
print(income/6)

