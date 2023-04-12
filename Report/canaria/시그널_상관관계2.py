# -*- coding: utf-8 -*-
"""
Created on Fri May 13 02:12:53 2022

@author: 82106
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  9 22:57:35 2022

@author: ajou
"""
#%%라이브러리 소환
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%데이터 소환
ism = pd.read_csv('C:\\Users\\82106\\Desktop\\DART\\레포트\\ism.csv',index_col=0).shift(1)
국채금리 = pd.read_csv('C:\\Users\\82106\\Desktop\\DART\\레포트\\국채금리.csv',index_col=0)
유가 = pd.read_csv('C:\\Users\\82106\\Desktop\\DART\\레포트\\유가.csv',index_col=0)
기준금리 = pd.read_csv('C:\\Users\\82106\\Desktop\\DART\\레포트\\한은_기준금리.csv',index_col=0)
환율 = pd.read_csv('C:\\Users\\82106\\Desktop\\DART\\레포트\\환율.csv',index_col=0)
코스피월간수익률 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\코스피.xlsx',index_col=0)

경상수지 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\경상수지.xlsx',index_col=0).shift(1)
high_yield = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\high_yield.xlsx',index_col=0)
#m2증가율 = pd.read_excel('C:\\Users\\ajou\\Desktop\\국면인식\\경제지표\\m2증가율.xlsx',index_col=0)
빅스 = pd.read_excel('C:\\Users\\82106\\Desktop\\DART\\레포트\\VIX.xlsx',index_col=0)

환율.info()
환율['rate']= pd.to_numeric(환율['rate'])

국고채3년 = 국채금리.iloc[:,0]
국고채5년 = 국채금리.iloc[:,1]
국고채10년 = 국채금리.iloc[:,2]  #국고채 가중치문제

금리차 = 국고채10년-국고채3년

#%%거리 재기(test)

트렌드 = ism.iloc[-6:].squeeze().to_numpy()    #표준화거리: https://rfriend.tistory.com/201
트렌드_표준화 = (트렌드 - np.mean(트렌드))/(np.std(트렌드)*6)
list_dist = []
list_corr = []
for i in range(1,len(환율)-6):
    trend = ism.iloc[-6-i:-i].squeeze().to_numpy()
    trend_standard = (trend - np.mean(trend))/(np.std(trend)*6)
    distance = ((트렌드_표준화 - trend_standard)**2).sum()**(1/2)
    list_dist.append(distance)
    
    corr = np.corrcoef(trend,트렌드)[0,1]
    list_corr.append(corr)
  
    
distance = np.array(list_dist)
correlation = np.array(list_corr)
model = distance + 1 - correlation

xx = pd.DataFrame(model)
국면인식 = xx.sort_values(by=0).iloc[:12]

#%%거리 재기

데이터프레임 = pd.DataFrame(model)  #래깅필요
dumdum = [ism,유가, 환율, 금리차, high_yield, 빅스, 경상수지]
for data in dumdum:
    트렌드 = data.iloc[-6:].squeeze().to_numpy()
    트렌드_표준화 = (트렌드 - np.mean(트렌드))/(np.std(트렌드)*6)
    list_dist = []
    list_corr = []
    for i in range(1,len(환율)-6):
        trend = data.iloc[-6-i:-i].squeeze().to_numpy()
        trend_standard = (trend - np.mean(trend))/(np.std(trend)*6)
        distance = ((트렌드_표준화 - trend_standard)**2).sum()**(1/2)
        list_dist.append(distance)
        
        corr = np.corrcoef(trend,트렌드)[0,1]
        list_corr.append(corr)
    distance = np.array(list_dist)
    correlation = np.array(list_corr)
    model = distance + 1 - correlation
    yy = pd.DataFrame(model)
    데이터프레임 = pd.concat([데이터프레임,yy],axis=1)  
    
데이터프레임 = 데이터프레임.iloc[:,1:]
데이터프레임['평균'] = 데이터프레임.T.sum()/len(dumdum)

#데이터프레임.columns = ['ism', '유가', '환율', '국고채3년','국고채5년','국고채10년', 'high_yield, "VIX", "평균"]

#%%데이터 처리

역순 = ism.iloc[list(range(ism.shape[0]-1,-1, -1))].index
역순_test = 역순[:len(환율)-6-1]
역순_test1 = 역순[1:len(환율)-6]
#df = pd.DataFrame(역순).iloc[:98]
df_df = pd.DataFrame([역순_test1,데이터프레임['평균']], index = ['날짜', '평균']).T
df_df = df_df.set_index('날짜')
국면인식_날짜 = df_df.sort_values(by='평균').iloc[:sort_values(by='평균').iloc[:12]12]


new_index = 코스피월간수익률.index[6:-1][::-1]
test_df = df_df
test_df['datetime'] = new_index 
test_df = test_df.iloc[3:]
test_df = test_df.set_index('datetime')  #코스피월간 수익률과 동일한 index form 만들기(date time)
datete_time_국면인식 = test_df.sort_values(by='평균').iloc[:12]

#%% n개월 수익률 구하기(cumprod) #query메소드: https://m.blog.naver.com/wideeyed/221867273249
코스피월간수익률_new = 코스피월간수익률.reset_index()
코스피월간수익률_new['변환'] = (코스피월간수익률_new['코스피']+100)/100

기하 = 코스피월간수익률_new[코스피월간수익률_new['Date'] == datete_time_국면인식.index[0]].index[0]     #for문 변수 index[i]
코스피월간수익률_new['변환'].iloc[기하:기하+3].cumprod()[기하+2]                                          #for문 변수 기하+3


빈리스트 = []
기하수익률리스트 = []
for i in range(0,12):
    기하 = 코스피월간수익률_new[코스피월간수익률_new['Date'] == datete_time_국면인식.index[i]].index[0]
    기하수익률 = 코스피월간수익률_new['변환'].iloc[기하:기하+3].cumprod()[기하+2]
    #딕셔너리 = '{}의 3개월 기하수익률{}'.format(datete_time_국면인식.index[i],기하수익률)
    #빈리스트.append(딕셔너리)
    기하수익률리스트.append(기하수익률.round(3))
완성 = pd.DataFrame(기하수익률리스트, index=datete_time_국면인식.index)
완성.columns = ['3개월 기하평균 수익률']


    
산술평균수익률 = np.array(기하수익률리스트).mean()
print(산술평균수익률)

#%%플롯
plt.figure(figsize=(20,6)) #데이터프레임 칼럼 바꿔야함
plt.plot(데이터프레임.iloc[:,0:6],alpha=0.1)
plt.plot(데이터프레임['평균'])

#%%백테시작해보자
#

ism.shift(1).iloc[1:]
경상수지.shift(1)
ism
경상수지

#%% 불베어


n=3
코스피기간별수익률 = 코스피월간수익률.copy()
코스피기간별수익률['코스피'] = (코스피기간별수익률['코스피']+100)/100
코스피기간별수익률['누적수익률'] = 0

#기간 전체의 수익률 데이터 프레임 생성 -> 코스피 기간별 수익률
for i in range(1+n, len(코스피기간별수익률)):
    코스피기간별수익률['누적수익률'].iloc[i] = 코스피기간별수익률['코스피'].iloc[i-n:i].cumprod()[2]-1

#거리구하면서 6개월썼으니 인덱스 맞춰주기
코스피기간별수익률=코스피기간별수익률.iloc[6:-1]

#train / test 구분
trainset = 코스피기간별수익률.iloc[:109,: ]
testset = 코스피기간별수익률.iloc[109:,: ]

#train/test 각각 하락/횡보/상승장 판독 -> iqr로 일단 기준설정
plt.hist(trainset['누적수익률'], bins = 20, cumulative=False)
plt.show()
trainset['누적수익률'].describe()
trainset['누적수익률'].skew()
plt.hist(testset['누적수익률'], bins = 20, cumulative=False)
plt.show()
testset['누적수익률'].describe()
testset['누적수익률'].skew()

bear_train = trainset[trainset['누적수익률'] <= -0.001428].index
bull_train = trainset[trainset['누적수익률'] >= 0.031659].index
bear_test = testset[testset['누적수익률'] <= -0.011508].index
bull_test = testset[testset['누적수익률'] >= 0.016815].index

#후에 쓸일이 있습니다. 224개의 인덱스
index_df = pd.DataFrame(코스피월간수익률.index)

#시장종류, dateindex를 넣어서 해당 시점과 비슷한 시점들 색출-> 
#그 지점들이 동일한 분류(에 들어가 있는지 확인해주는 함수
def eval(market, dateindex) :
    dumdum = [ism,유가, 환율, 금리차, high_yield, 빅스, 경상수지]
    #파라미터 날짜에 해당하는 데이터 인덱스 색출
    num = index_df.index[index_df['Date'] == dateindex][0]
    #반복문으로 각 지표별 6개월간의 데이터 뽑아낼건데 역순으로
    for data in dumdum:
        #해당 데이트인덱스로 부터 6개월간의 데이터 뽑아내
        트렌드 = data.iloc[num-6:num].squeeze().to_numpy()
        트렌드_표준화_f = (트렌드 - np.mean(트렌드))/(np.std(트렌드)*6)
        list_dist = []
        list_corr = []
        for i in range(1,len(환율)-6):
            trend = data.iloc[-7-i:-1-i].squeeze().to_numpy()
            trend_standard = (trend - np.mean(trend))/(np.std(trend)*6)
            distance = ((트렌드_표준화_f - trend_standard)**2).sum()**(1/2)
            list_dist.append(distance)
        
            corr = np.corrcoef(trend,트렌드)[0,1]
            list_corr.append(corr)

        distance = np.array(list_dist)
        correlation = np.array(list_corr) 
        
        model = distance + 1 - correlation
        model_test = model[:109]
        model_train = model[109:]
        
        yy = pd.DataFrame(model_train)
        zz = pd.DataFrame(model_test)
        
        데이터프레임_train = pd.DataFrame(model_train)
        데이터프레임_test = pd.DataFrame(model_test)
        데이터프레임_train = pd.concat([데이터프레임_train,yy],axis=1)
        데이터프레임_test = pd.concat([데이터프레임_test,zz],axis=1)
        
    데이터프레임_train = 데이터프레임_train.iloc[:,1:]
    데이터프레임_test = 데이터프레임_test.iloc[:,1:]
    
    데이터프레임_train['평균'] = 데이터프레임_train.T.sum()/4
    데이터프레임_test['평균'] = 데이터프레임_test.T.sum()/4


    역순 = ism.iloc[list(range(ism.shape[0]-1,-1, -1))].index
    역순_index = 역순[1:len(환율)-7]
    역순_test = 역순_index[:108]
    역순_train = 역순_index[108:]
    df_df = pd.DataFrame([역순_train,데이터프레임_train['평균']], index = ['날짜', '평균']).T
    df_df = df_df.set_index('날짜')
    국면인식_날짜 = df_df.sort_values(by='평균').iloc[:12]




    acc = 0 
    for i in range(0,12):        
         acc += int(국면인식_날짜.index[i] in market)   
    return acc/12
    


bear_acc = 0
for i in range(27):
    bear_acc += eval(bear_train, bear_test[i])
bear_acc = bear_acc/27
bear_acc

bull_acc = 0
for i in range(27):
    bull_acc += eval(bull_train, bull_test[i]) 
bull_acc = bull_acc/27
bull_acc

eval(bear_train, bear_test[0])







#%% 상관관계



list_train = []
list_test = []

#시장종류, dateindex를 넣어서 해당 시점과 비슷한 시점들 색출-> 
#그 지점들이 동일한 분류(에 들어가 있는지 확인해주는 함수
def correal(dateindex) :
    list_test.append(코스피기간별수익률['누적수익률'][dateindex])
    dumdum = [ism,유가, 환율, 금리차, high_yield, 빅스, 경상수지]
    #파라미터 날짜에 해당하는 데이터 인덱스 색출
    num = index_df.index[index_df['Date'] == dateindex][0]
    
    for data in dumdum:
        #해당 데이트인덱스로 부터 6개월간의 데이터 뽑아내
        트렌드 = data.iloc[num-6:num].squeeze().to_numpy()
        트렌드_표준화_f = (트렌드 - np.mean(트렌드))/(np.std(트렌드)*6)
        list_dist = []
        list_corr = []
        for i in range(1,len(환율)-6):
            trend = data.iloc[-6-i:-i].squeeze().to_numpy()
            trend_standard = (trend - np.mean(trend))/(np.std(trend)*6)
            distance = ((트렌드_표준화_f - trend_standard)**2).sum()**(1/2)
            list_dist.append(distance)
        
            corr = np.corrcoef(trend,트렌드)[0,1]
            list_corr.append(corr)

        distance = np.array(list_dist)
        correlation = np.array(list_corr) 
        
        model = distance + 1 - correlation
        model_test = model[:109]
        model_train = model[109:]
        
        yy = pd.DataFrame(model_train)
        zz = pd.DataFrame(model_test)
        
        데이터프레임_train = pd.DataFrame(model_train)
        데이터프레임_test = pd.DataFrame(model_test)
        데이터프레임_train = pd.concat([데이터프레임_train,yy],axis=1)
        데이터프레임_test = pd.concat([데이터프레임_test,zz],axis=1)
        
    데이터프레임_train = 데이터프레임_train.iloc[:,1:]
    데이터프레임_test = 데이터프레임_test.iloc[:,1:]
    
    데이터프레임_train['평균'] = 데이터프레임_train.T.sum()/4
    데이터프레임_test['평균'] = 데이터프레임_test.T.sum()/4


    역순 = ism.iloc[list(range(ism.shape[0]-1,-1, -1))].index
    역순_index = 역순[1:len(환율)-7]
    역순_test = 역순_index[:108]
    역순_train = 역순_index[108:]
    df_df = pd.DataFrame([역순_train,데이터프레임_train['평균']], index = ['날짜', '평균']).T
    df_df = df_df.set_index('날짜')
    국면인식_날짜 = df_df.sort_values(by='평균').iloc[:1]




    acc = 0 
    for i in range(0,12):        
         acc+=코스피기간별수익률['누적수익률'][국면인식_날짜.index[i]]  
    list_train.append(acc/12)


for i in range(108):
    correal(testset.index[i])
    


plt.hist(list_test)
plt.hist(list_train)

plt.plot(list_test,list_train,"or")
plt.ylabel('list_train')
plt.xlabel('list_test')
plt.title('correlation')
np.corrcoef(list_train, list_test)[0,1]
