# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:31:21 2023

@author: Ehsan
"""

import wrds
import pandas as pd 
from pandas_datareader import data
import pandas as pd
import pandas_datareader as web
import yfinance as yf
import yesg

import pandas as pd
from nueramic_mathml import HiddenPrints
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import plotly.express as px
from tqdm import tqdm
from plotly.subplots import make_subplots
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler


###########  get the stock symboles
from stocksymbol import StockSymbol
# get symbol list based on market
api_key = '55111e3d-74a0-44d3-9813-de03aa1a010d'
ss = StockSymbol(api_key)
symbol_list_us = ss.get_symbol_list(market="US") 

# get symbol list based on index
symbol_list_dow = ss.get_symbol_list(index="DJI")
symbol_list_SP500 = ss.get_symbol_list(index="SPX")
df1 = pd.DataFrame(symbol_list_SP500)
df2 = df1['symbol']
df2 = df2.tolist()


############################ import the ESG data from WRDS
db = wrds.Connection()
db.list_libraries()
db.list_tables(library='sustainalytics_all')
stocknames1 = db.get_table(library='sustainalytics_all', table='hist_weighted_score')
stocknames2 = db.get_table(library='sustainalytics_all', table='hist_weighted_score', obs=10)

esgdata = stocknames1[['date', 'company', 'peer_group_root', 'region', 'country', 'ticker', 'total_esg_score',
                        'social_score', 'governance_score', 'environment_score']]
esgdata.head(10)
esgdata = esgdata.replace(to_replace='None', value=np.nan).dropna()

#esgdata1 = esgdata [(esgdata['region'] == 'United States and Canada') or (esgdata['region'] == 'North America')]
esgdata1 = esgdata [ (esgdata['country'] == 'United States')]
#esgdata1 = esgdata [(esgdata['region'] == 'Europe')]

############################  merge ESG data and stock price data from yahoo finance
def download_data1(ticker: str) -> [pd.DataFrame, None]:
       
    esg = esgdata1[esgdata1['ticker'] == ticker]
    esg['date'] = pd.to_datetime(esg['date']).dt.strftime('%Y-%m') 
    if esg is not None:
        if esg.shape[0] > 10:
            #start_date = pd.Timestamp(esg.index.min())
            #end_date = pd.Timestamp(esg.index.max())
            start_date = pd.Timestamp('2009-09-01 00:00:00')
            end_date = pd.Timestamp('2018-02-01 00:00:00')

            stock = yf.download(ticker, start=start_date, end=end_date, interval='1mo')
            stock ['pct'] = stock['Adj Close'].pct_change()# Creating daily percentage change in one month (interval='1mo') period
            stock ['cum_return'] = (1 + stock ['pct']).cumprod() - 1
            a = []
            for row in stock ['pct']:
                if row < -0.05:
                    a.append(1)
                else:
                    a.append(0)
            stock ['crash'] = a
            if stock is not None:
                esg['date'] = pd.to_datetime(esg['date']).dt.strftime('%Y-%m')
                stock.reset_index(inplace=True)
                stock['Date'] = pd.to_datetime(stock['Date']).dt.strftime('%Y-%m')
                stock.rename(columns={'Date':'date'}, inplace=True)
                stock = stock.drop(['Open', 'Low', 'High', 'Volume', 'Close'], axis=1)
                out = pd.merge(esg, stock)
                
                return out.dropna()
        
        else:
            return None


full_tickers = df2
########################  extract the return of benchmark index sp500
start_date = pd.Timestamp('2009-09-01 00:00:00')
end_date = pd.Timestamp('2018-02-01 00:00:00')
sp500 = yf.download('SPY', start=start_date, end=end_date, interval='1mo')
sp500['pct'] = sp500['Adj Close'].pct_change()




######################## cross sectinal analysis
full_data = pd.DataFrame(columns = ['governance_score', 'social_score','environment_score','total_esg_score',
                         'pct.mean', 'pct.volatiltiy', 'max_drawdown','beta', 'alpha', 'marketCap' ] )

for ticker in tqdm(full_tickers):
    
    df = download_data1(ticker)
    if df is None:
        pass
    else:
        if df.shape[0]>20:
            size = sp500.shape[0]-df.shape[0]
            data1 = []
            data1.append(df['governance_score'].mean())
            data1.append(df['social_score'].mean())
            data1.append(df['environment_score'].mean())
            data1.append(df['total_esg_score'].mean())
            data1.append(df['pct'].mean())
            data1.append(np.std(df['pct']))
            data1.append(abs(pf.timeseries.max_drawdown(df['pct'])))
            data1.append(np.polyfit(df['pct'], sp500['pct'][size:], 1)[0])
            data1.append(np.polyfit(df['pct'], sp500['pct'][size:], 1)[1])
            data1.append(web.get_quote_yahoo(ticker)['marketCap'][0])
            
            full_data.loc[len(full_data.index)] = data1 
            
           
                        
# Check for multicollinearity using correlation plot
corr = full_data[[ 'governance_score', 'social_score','environment_score','total_esg_score',
                         'pct.mean', 'pct.volatiltiy', 'max_drawdown','beta', 'alpha', 'marketCap']].corr()
f,ax = plt.subplots(figsize=(20,12))
sns.heatmap(corr, annot=True, cmap='Reds', linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


####### Ridge/ linear regression 
x, y = full_data[['governance_score', 'social_score','environment_score','total_esg_score',
                          'pct.volatiltiy', 'max_drawdown','beta', 'alpha', 'marketCap']], full_data['pct.mean']    
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
#lm = LinearRegression().fit(X_train, y_train)
lm = Ridge().fit(X_train, y_train)
rsquared = r2_score(y, lm.predict(x))
metrics.mean_squared_error(y, lm.predict(x))
coeficients = lm.coef_ 
           
#######   random forest regression            
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
rf = RandomForestRegressor(n_estimators=100, max_features=9, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
#accuracy=metrics.recall_score(y_test, y_pred, pos_label="no")
CV_score=cross_val_score(rf, X_train, y_train, cv=5).mean()
print("=> test CV_score= %.3f" % CV_score)




####################    time series analysis for each stock
data = []   
for ticker in tqdm(full_tickers):
    df = download_data1(ticker)
    if df is None:
        pass
    else:
        if df.shape[0] > 5:
            x, y = df[['total_esg_score','governance_score', 'social_score', 'environment_score', 'pct' ]], df['crash']  
            X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
            rf = RandomForestClassifier(n_estimators=100,  random_state=0)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            #accuracy=metrics.recall_score(y_test, y_pred, pos_label="no")
            CV_score=cross_val_score(rf, X_train, y_train, cv=5).mean()
            print("=> test CV_score= %.3f" % CV_score)
            data.append(CV_score)


data = []
coefs = []
for ticker in tqdm(full_tickers):
    
    df = download_data1(ticker)
    if df is None:
        pass
    else:
        if df.shape[0]>20:
            
            if df.shape[0] != 0:
                x, y = df[['total_esg_score','governance_score', 'social_score', 'environment_score']], df['pct']
                #x, y = df['Total-Score'], df['pct']
                X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
                lm = Ridge().fit(X_train, y_train)
                #lm = LinearRegression().fit(X_train, y_train)
                #lm = LinearRegression().fit(X_train, y_train)
                print("Train-R^2 of linear regression: %.3f" % lm.score(X_train, y_train))
                print("Test-R^2 of linear regression: %.3f" % lm.score(X_test, y_test))
                data.append(r2_score(y, lm.predict(x)))
                coefs.append(lm.coef_)

# show feature importance
importances = list(rf.feature_importances_)
feature_importances = [(feature, importance) for feature, importance in zip(X_train.columns.values, importances)]
for f,i in feature_importances:
    print("Feature: %s Importance: %.3f" % (f, i))
    
# plot feature importance
fig=plt.figure(figsize=(16, 6))
plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_, align="center")
plt.xticks(range(len(X_train.columns.values)), X_train.columns.values, rotation="vertical")
plt.title("Feature importance")
plt.ylabel("Importance")
plt.xlabel("Features")
fig.subplots_adjust(bottom=0.3) 
plt.savefig("feature_importance.pdf")
plt.show()
            