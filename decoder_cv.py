#code for running naive bayes classifier on binned unit data
#
#NOTE: this code runs off data that is output by the main encoder models
#
#DSJ 2022

#oackages
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
from glob import glob
from plotnine import *

from random import randint


# if we use scikit
from sklearn.datasets import load_iris #dont actually need this
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sllearn.model_selection import KFold



#data directory
os.chdir('matrices')
#get data (both a matrix of trial conduntions and processed single units)
files=glob('*S2*')
df=pd.read_csv('trialmat.csv')
getwindows=pd.read_csv(files[0]).drop(columns=['Unnamed: 0']).columns
#scaler for normalizing values
scaler = StandardScaler()

#set up 
res=[]
shuf=[]
semres=[]
semshuf=[]


#aggregate data for each timepoint for each unit
# this could be stored in a better way to that you dont need seperate files for each unit. They are intentially written this way per request
for timepoint in getwindows:
	window=timepoint
	forclassi=pd.DataFrame()
	for a in files:
		df2=pd.read_csv(a)
		df2=df2.drop(columns=['Unnamed: 0'])
		forclassi[a]=df2[window]
		
#get and normalize data
	xlist=forclassi.columns
	X=forclassi.to_numpy()
	Xscale=scaler.fit_transform(X)
	#to be predicited (classification of the US)
	y=df['US'].to_numpy()
	#shuffled version
	shufout=pd.Series(y).sample(frac=1).to_numpy()

	#fir storing results
	result=[]
	shuffle=[]
	resSTD=[]
	shufSTD=[]

	# the real work starts here. In a past version I didnt use kfold CV and just re ran the analysis several times as indicated by the value in range().
	# since we use kfold CV we just rin this once
	for a in range(1):
		#use naiveBayes and 10-fold CV
		gnb = GaussianNB()
		kf=KFold(n_splits=10,shuffle=True)
		score=cross_val_score(gnb,Xscale,y,cv=kf,scoring='accuracy')
		#save the result
		accuracy=np.mean(score)#outcomes.sum()/len(outcomes)
		err=np.std(score)/np.sqrt(10)
		result.append(accuracy)
		resSTD.append(err)
	#now do it with outcome shuffled data
		gnb = GaussianNB()
		kf=KFold(n_splits=10,shuffle=True)
		shufscore=cross_val_score(gnb,Xscale,shufout,cv=kf,scoring='accuracy')

		shufaccuracy=np.mean(shufscore)#shuf_outcomes.sum()/len(shuf_outcomes)
		shuferr=np.std(score)/np.sqrt(10)
		shuffle.append(shufaccuracy)
		shufSTD.append(shuferr)
	res.append(np.mean(result))
	shuf.append(np.mean(shuffle))
	semres.append(resSTD[0])#/np.sqrt(a+1))
	semshuf.append(shufSTD[0])#/np.sqrt(a+1))


# quick  plot of trial time by decoding accuracy. Chance is 50% in this case,,,
t=np.array(range(len(res)))
res=np.array(res)
shuf=np.array(shuf)
semres=np.array(semres)
semshuf=np.array(semshuf)
plt.plot(res,color='#3498DB',linewidth=3)
#plt.plot(USres,color='#DC7633',linewidth=3)
plt.plot(shuf,color='#AEB6BF',alpha=.8,linewidth=3)
plt.fill_between(t,res+semres,res-semres,facecolor='#3498DB',alpha=.5,zorder=2)
plt.fill_between(t,shuf+semshuf,shuf-semshuf,facecolor='#AEB6BF',alpha=.5,zorder=2)
plt.axvline(x=39,color='black')
plt.axhline(y=.5,color='grey',linestyle='dashed')
plt.xticks(ticks=[19,39,59],labels=['-1','0','1'])
plt.ylabel('Decoding Accuracy')
plt.xlabel('Peri CS Onset')
plt.ylim(.40,.70)
plt.legend(['US','Shuffle'])
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
#plt.ylim(.3,1)
plt.show()









