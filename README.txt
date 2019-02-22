Our group name: Rocket 101


Goal:
Our project aims to build a model to find the relationshiop between S&P500 stock index change and daily financial news, and to make predictions.


Overview:
We scrape daily financial news and stock price data from websites. We execute the text mining using some inner packages including NLTK. We use two methods to calculate market sentiment scores, one is SentimentIntensityAnalyzer, and the other is TextBlob.
Then, we process dependent variable -- S&P500 price. First, we calculate daily growth based on change of adjusted close price and open price. Second, we generate LabeledY, which is composed of -1, 0, 1 to indicate the direction of market change(negative, neutral, positive). 
After data processing, we use several models to fit our dataset, including Simple Linear Regression Model, Decision Tree Model, Supported Vector Regression Model(SVR), Supported Vector Classification Model(SVC) and XGBoost Model.
In the process, we find that Simple Linear Regression Model has a low R square and does not fit our dataset. Decision Tree Model fits the train data well, however, it does not fit the test data, and we consider this model overfitting. Besides, the SVC model is a fair fit for the classification of dataset while SVR is not fitting since the accuracy is too low whether in "linear", "polynomial" or "rbf" kernel. What's more, the XGBoost Model behaves better with 84.67% accuracy in train sample, and 40.52% accuracy in test sample.


Python Library:
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from pandas_datareader import data as web
import os
import string
import requests
import nltk
from nltk import word_tokenize
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import seaborn
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from wordcloud import WordCloud
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score


Procedure:
First: Web Scraping Data
1. Scraping stock price from yahoo finance by using BeautifulSoup, get S&p500 Index from Sept.26 2016 to Oct.1 2018.
2. Scraping news title and news abstract from Wall Street Journal(http://www.wsj.com) by using BeautifulSoup.
3. Convert Scraping results into DataFrame using pandas, and write out .csv file as our project data.
4. Output File are news.csv and Stock_Price.csv


Second: Data virtualization
1. Data insight of Market Data using Plotly from several dimensions.
2. Data insight of News Data using Plotly to get a direct impression of news headlines and abstract.


Third: News data Textmining
1. Filter Text by using NLTK  and many inner-packages:
Remove digits, remove white space, remove stopwords, lemmatizer. Then write into .csv as "newscleaned1.csv"
2. Using sklearn package to count words frequency on both title and abstract, delete the low frequency words(<5) 
and re-orgnize text data file into data frame. Then wrote out .csv file as "newscleaned2.csv"


Fourth: Index Data Preprocessing
1. Calucate HighMinusLow and CloseMinusOpen based on daily stock price data.
2. Generate LabeledY according to 1-sigma and 0.5-sigma rule since market has reasonable daily volatility. Treat the daily growth as positive(+1)/negative(-1) when it is outside of the range, eg.(mean-0.5sigma,mean+0.5sigma).


Fifth: Text segmentation and Polarity Analysis
1. Using Textblob to do this step. Convert both news title and news abstract to two-dimensional columns
polarity and subjective. Then our explantory varible X formed. 
2. Using SentimentIntensityAnalyzer in NLTK to do this step. We assign polarity scores to each day's financial news titile and financial news abstract.


Sixth: Partition Training and Test
1. Using the first 70% data as train, and last 30% data as test. Using this partition on both X and Y.
2. Other cutting methods


Seventh: Modelling
1. Using Simple Linear Regression to train our model. However, it has negative R square, which indicates this model is not fitted.
2. Using Decision Tree Model to train our model. Our train set shows high accuracy, while our test set demonstrates low performance.
3. Using SVC(support vector classification) to train and test our model, it has high performance in training and fair performance in testing.
4. Using SVR(support vector regression）to train our model, but it has depressing performance thus it's not a fit neither.
4. USing XGBoost to train our model. After tuning, eventaully get test accuracy: 0.8507042253521127.


Eighth: Result analysis
1.For Linear Regression Model, our data does not have linear relationshiop and this model is not good.
2.For Decision Tree Model, the train data generates a good fitting model, however, the test data does not fit the model well. So we can not apply this model for future usage.
3.For SVC, it does a fair classification of our x and y with testing accuary of approximately 58%.
4.For SVR, it's not a fit for our dataset since it has low accuracy whether in "linear","polynomial" or "rbf" kernel.
5.For XGBoost Model, it has 84.67% accuracy in train sample, and 40.52% accuracy in test sample. 
Considering all the method we have tried, we think XGBoost Model is best fit for our dataset.


Run Instruction:
Our project is a model building process regarding finance area, you just need to run the code as the procedure sequence indicates. 

