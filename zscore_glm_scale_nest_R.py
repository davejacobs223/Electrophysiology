#
#Code for doing the heavy analysis of unit data. it will do three things from the pickle DFs written out by mat2py
#1) build a trial matirx which contains the stimuli and outcomes for each trial
#2) take unit data to build an encoding model to see which aspect of the task is or is not encoded by a unit
#3)make some helpful plots for determining if a unit is responsive at a particular ime, stable over time, or is  likely multi unit activity
#
#NOTE: this code requires R. It uses rpy2 to port into R so it can take advantage of R statistical models when it fits the GLM. I originally wrote it for
#python's statsmodels but decided to switch to R due to the way it handled covariance matrix issues and nested models.
#written DSJ 2022


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
from glob import glob
from plotnine import *
#for AOV
import statsmodels.api as sm
from statsmodels.formula.api import ols
# porting in to R with rpy
import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


utils = importr('utils')
base = importr('base')
stats = importr('stats')
car = importr('car')


bigboi=pd.DataFrame()
logps=pd.DataFrame()

metrics=[]

#DECODE THE EVENTS
event=pd.read_csv('events.csv')
#plot p vals?
plotting=True
#Specify session type so the stim codes are right (TO BE ADDED)
if 'aaaa'=='aaaa':
	stimcodres=[]

on=[]
off=[]
#get trial start aand stop locations - 9,18
for a in range(len(event)):
	if event['Marker'][a] == 9.0 and a < 2:
		on.append(a)
	elif event['Marker'][a] != 9.0 and a < 2:
		pass
	elif event['Marker'][a] == 9.0 and event['Marker'][a-1] == 18.0:
		on.append(a)
	elif event['Marker'][a] == 2.0 and a == 0:
		on.append(a)
		print('2block')
	elif event['Marker'][a] == 18.0 and event['Marker'][a-2] != 9.0:
		off.append(a)

#get CS/US info for each trial
master=[]
for a in range(len(on)):
	start=on[a]
	stop=off[a]

	trialspan=event['Marker'][start:stop+1]
#trialcount= len(event[event['Marker']==9]) #there are extra 0s which is weird...
	if len(trialspan.isin([8,14,26]).value_counts())>1:
		if trialspan.iloc[np.where(trialspan.isin([8,14,26])==True)[-1][0]+1]==18.0:
			US=1
			UStype=trialspan.iloc[np.where(trialspan.isin([8,14,26])==True)[-1][0]]
		else:
			US=0
			UStype=8675309
	else:
		US=0
		UStype=8675309

	if len(trialspan.isin([11,12]).value_counts())>1:
		if trialspan.isin([11,12]).value_counts()[True]>5:
			CS=0
			CStype=trialspan.iloc[-3]
		else:
			CS=1
			CStype=trialspan.iloc[-3]
	elif len(trialspan.isin([21,22]).value_counts())>1:
		if trialspan.isin([21,22]).value_counts()[True]>5:
			CS=1
			CStype=trialspan.iloc[-3]
		else:
			CS=0
			CStype=trialspan.iloc[-3]
	else:
		print('?????')

#get stim trials and blocks in there

	#blockid=
	place=1#np.where(trialspan==9)[0][0]+1
	block=trialspan.to_numpy()[place]
	#stimid
	splace=np.where(trialspan==block)[0][0]+1
	stimid=trialspan.to_numpy()[splace]

	abstrial=a+1
#save data
	master.append([abstrial,CS,US,stimid,block,CStype,UStype])


#make array, get unique stims
master=np.array(master)
stims=np.unique(master[:,3])
#i dont think thes line here is used....
blocks=np.unique(master[:,4])


#for loop to count the stim trials
main=np.empty(8)
for a in np.unique(master[:,3]):
	substim=master[master[:,3]==a]
	span=len(substim)
	counts=np.linspace(1,span,span)
	out=np.column_stack((substim, counts))
	main=np.vstack((main,out))

main=main[1:]


#main matrix of trial conditions
trialmat=pd.DataFrame(main).sort_values(by=[0]).rename(columns={0:'ABStrial',1:'CS',2:'US',3:'stimid',4:'block',5:'CStype',6:'UStype',7:'stimtrial'})

#recode the blocks as 1,2,3...
border=trialmat[trialmat['block'].diff()!=0]
border['bid']=list(range(1,len(border)+1,1))
border=border[['block','bid']].set_index('bid')#.to_dict()['block']

blocksused=[]
forDF=[]
priorblock=border.iloc[0][0]
for item in range(len(trialmat)):
	rawblock=trialmat.iloc[item]['block']
	if rawblock != priorblock:
		blocksused.append(priorblock)
	else:
		pass
	if rawblock in blocksused:# this is a cheap fix for when we duplicate blocks. if we use a block more then twice we will have to better write this
		coded_block=border[border['block']==rawblock].index[1]
	else:
		coded_block=border[border['block']==rawblock].index[0]
	priorblock=rawblock
	forDF.append(coded_block)


#last minute adds and resturcturing for future GLM
trialmat['stimid']=trialmat['stimid'].astype('str')
trialmat['block']=trialmat['block'].astype('str')
trialmat['block_recode']=forDF

#get p for each stim
newtrialmatblock=pd.DataFrame()
for cs in trialmat['CStype'].unique():
	df=trialmat[trialmat['CStype']==cs]


	blockdf=pd.DataFrame()
	#for a block
	for ablock in df['block'].unique():
		blockeddf=df[df['block']==ablock]
		meanlist=[]
		for a in range(len(blockeddf['US'])):
			if a == 0:
				init=[.5]+blockeddf['US'][0:a].tolist()   # currently set to a priori probability. to make a posterioiri it would be [0:a+1], sum(init)+1 here and [0:a+1] in both places on line 153 below
				meanlist.append(sum(init))
			else:
				meanlist.append((blockeddf['US'][0:a].sum()+.5)/(len(blockeddf['US'][0:a])+1))
		blockeddf['Prob']=meanlist
		blockdf=blockdf.append(blockeddf)

	blockdf['tord']=list(range(1,len(blockdf)+1,1))

	newtrialmatblock=newtrialmatblock.append(blockdf)
newtrialmatblock=newtrialmatblock.sort_values(by=['ABStrial'])



#save out 
#output=pd.DataFrame(main).sort_values(by=[0]).rename(columns={0:'ABStrial',1:'CS',2:'US',3:'stimid',4:'block',5:'stimtrial'})
regions=['Mod4','Mod8']
allstuff=glob('*')
folderstodo=[a for a in allstuff if a in regions]
#folderstodo=[folderstodo[1:3]]
#folderstodo=['L_STR','R_STR','L_V2']#L_AMY
#folderstodo=['R-AMY']



trialmat2=newtrialmatblock
expid=os.getcwd().split('\\')[-1] 


#rodeo codes:
#if os.getcwd().split('\\')[-1] == 'flickerpav': 


#this is where we move from main directory
if not os.path.exists('matrices'):
	os.makedirs('matrices')
os.chdir('matrices')
trialmat2.to_csv('trialmat.csv')
os.chdir('../')


for reg in folderstodo:
	countunits=0
	os.chdir(reg)


	channelstoanalyze=glob('pin*')
	#channelstoanalyze=[channelstoanalyze[1]]

	for chan in channelstoanalyze:
		os.chdir(chan)

	# binning and zscoring

		with open('unit_dictionary.pkl', 'rb') as f:
		   	trialdict = pickle.load(f)



		for a in trialdict.keys():
		#for a in [list(trialdict.keys())[1]]:
			if a ==0.0:
				print('skip')
				continue
			print(a)
		#pick out a unit
			countunits=countunits+1
			unitID=chan+'_'+str(a)
			trialdictunit=trialdict[a]
			unitfig='unit'+str(a)+'_pvals'+'.png'
			unitname=str(a)

			n_windows=int(4/.05) # 50 msec # make vari
			windowsize=.250 #.2 = 200 msec  # make var

			superlist=[]


			for trial in trialdictunit.keys():
				datatrace=np.array(trialdictunit[trial])
				saveme=[]

				for a in range(n_windows-3):
					norm=a*.05
					start=-2+norm # make sure this works right
					end=start+windowsize
					SR=((start <= datatrace) & (datatrace < end)).sum()/windowsize
					saveme.append(SR)

				#save=np.array(saveme)
				#znorm=np.abs((save-np.mean(save[60:81]))/(np.std(save[60:81])+.000001))
				superlist.append(saveme)

			new=np.array(superlist)


			#for peri-event average plot
			trialav=np.mean(new,axis=0)
			znorm=np.abs((trialav-np.mean(trialav[30:41]))/(np.std(trialav[30:41])))

		#Run the ANOVA (all IVs are seen as categorical, this is fair for all but stim trial which is continuous so we could switch to a GLM with dummy coding but meh)

			#drop last trial fro this file

			os.chdir('../../matrices')
			pd.DataFrame(new).to_csv('ymat'+reg+'_'+chan+'_'+unitname+'.csv')
			os.chdir('../'+reg+'/'+chan)




			samps=len(new[1])

			CSlist=[]
			USlist=[]

			stimtriallist=[]
			Problist=[]

			CSbystimID=[]
			USbyusID=[]


			for a in range(samps):
			#pick bin
				matforlm=trialmat2.copy()
				matforlm['SC']=new[:,a]
				binnumb=a+1


			#convert to R
				with (ro.default_converter + pandas2ri.converter).context():
					dataf = ro.conversion.get_conversion().py2rpy(matforlm)
				#run model
				mod=stats.lm('SC~as.factor(CS)/as.factor(stimid) + as.factor(US)/as.factor(UStype)+Prob+stimtrial',data=dataf)
				if stats.deviance(mod)[0]<=1e-20: #if there is no deviance in the model, return a nan for p. this means these was no variance in the SC, aka non significant effect
					CS_p=US_p=stimtrial_p=prob_p=CSbystimID_p=USbyusID_p=np.nan
				else:
					aovtable=car.Anova(mod,type=2)
					#back to python
					with (ro.default_converter + pandas2ri.converter).context():
						aov_pd = ro.conversion.get_conversion().rpy2py(aovtable)
					CS_p=aov_pd.loc['as.factor(CS)']['Pr(>F)']
					US_p=aov_pd.loc['as.factor(US)']['Pr(>F)']

					CSbystimID_p=aov_pd.loc['as.factor(CS):as.factor(stimid)']['Pr(>F)']    
					USbyusID_p=aov_pd.loc['as.factor(US):as.factor(UStype)']['Pr(>F)']

					stimtrial_p=aov_pd.loc['stimtrial']['Pr(>F)']
					prob_p=aov_pd.loc['Prob']['Pr(>F)']


			#append them
				CSlist.append(CS_p)
				USlist.append(US_p)

				stimtriallist.append(stimtrial_p)
				Problist.append(prob_p)

				CSbystimID.append(CSbystimID_p)
				USbyusID.append(USbyusID_p)


			glmout=[CSlist,USlist,stimtriallist,Problist,CSbystimID,USbyusID]
			glmDF=pd.DataFrame(glmout).transpose().rename(columns={0:'CS',1:'US',2:'stimtrial',3:'Prob',4:'CS*ID',5:'US*USID'})
			allps=pd.DataFrame(np.where(glmDF <.05,1,0))
			allps.columns=glmDF.columns
			allps=allps.transpose()
			allps.insert(0,'Region',reg)
			allps.insert(1,'unitID',unitID)
			allps= allps.reset_index().rename(columns={'index':'factor'})
			glmDF=-np.log10(glmDF)

			logGLM=glmDF.copy()
			logGLM=logGLM.transpose()
			logGLM.insert(0,'Region',reg)
			logGLM.insert(1,'unitID',unitID)
			logGLM= logGLM.reset_index().rename(columns={'index':'factor'})



			#optional p value plot, set for 6 plots....
			#minimum p is .00015
			if plotting==True:

				#glmDF[glmDF<.00015]=.00015
				#glmDF=glmDF.replace(np.nan,1.0)
				plt.clf()
				fig, ax = plt.subplots(nrows=3,ncols=3)
				count=0
				maincount=0
				xplaces=[0,1,2]
				yplace=[0,1,2]
				for a in glmDF.columns:
					if maincount > 2 and maincount < 6:
						xplace=xplaces[1]
						count=maincount-3
					elif maincount > 5:
						xplace=xplaces[2]
						count=maincount-6
					else:
						xplace=xplaces[0]

					ax[xplace,yplace[count]].plot(glmDF.index+1,glmDF[a])
					ax[xplace,yplace[count]].plot(glmDF.index+1,glmDF[a])
					#ax[xplace,yplace[count]].set_yscale("log")
					ax[xplace,yplace[count]].axhline(y=.05,color='black')
					ax[xplace,yplace[count]].axvline(x=40,color='black')
					ax[xplace,yplace[count]].axvline(x=60,color='black')
					#ax[xplace,yplace[count]].set_ylim(.0001,1.2)
					ax[xplace,yplace[count]].set_xlim(0,80)
					ax[xplace,yplace[count]].set_title(a)
					count=count+1
					maincount=maincount+1
				plt.savefig(unitfig)
				plt.clf()
			else:
				pass

			bigboi=pd.concat([bigboi,allps])
			logps=pd.concat([logps,logGLM])
			bigboi=bigboi.sort_values(by=['factor'])
			logps=logps.sort_values(by=['factor'])
		print (reg,':',countunits)
		os.chdir('../')
		metrics.append([reg,countunits])
	os.chdir('../')



#get unit involvment %
threshold=5
encode=[]
for areg in bigboi['Region'].unique():
	subdata=bigboi[bigboi['Region']==areg]
	for unitID in subdata['unitID'].unique():
		subsubdata=subdata[subdata['unitID']==unitID].transpose()
		for columns in subsubdata.columns:
			counter=0
			encoded=0
			dset=subsubdata[columns][3:]
			for item in range(len(dset)):
				if dset[item]==1:
					counter=counter+1
					if counter >=threshold:
						encoded=1
				else:
					counter=0
			encode.append([areg,areg+'-'+unitID,encoded])

df=pd.DataFrame(encode)
aggr=df.groupby(by=[0,1]).sum()

aggr[aggr[2]>=1].reset_index().rename(columns={0:'region',1:'id',2:'factors_encoded'}).to_csv('encoderids.csv')
encoders=aggr[aggr[2]>=1].reset_index()[1].tolist()

nencode=aggr[aggr[2]>=1].reset_index().groupby(by=0).count()[1].reset_index()
nnull=aggr[aggr[2]==0].reset_index().groupby(by=0).count()[1].reset_index()
nencode['nulls']=nnull[1]
nencode['total']=nencode['nulls']+nencode[1]
nencode['perc']=nencode[1]/nencode['total']
nencode=nencode.rename(columns={0:'region',1:'encoders'})
nencode.to_csv('summarystats'+'_threshold'+str(threshold)+'.csv')



#pretty plots
#prep the data
bigboi['both']=bigboi['Region']+'-'+bigboi['unitID']
bigboi2=bigboi[bigboi['both'].isin(encoders)]
bigboi2=bigboi2.drop(columns=['both'])

avgs=bigboi2.groupby(by=['factor','Region']).mean().reset_index()
#avgs=avgs[avgs['factor']!='CS_nest']
avgsr=pd.melt(avgs, id_vars=['factor', 'Region'], value_vars=list(range(0,77)))
avgsr['variable']=avgsr['variable'].astype(int)


percplot=(ggplot(avgsr,aes('variable','value',group='factor',color='factor'))
+geom_line(size=1)
+facet_wrap('~Region',scales='free_y')
+xlab('Time')
+ylab('prop Units (encoders only)')
+xlim(20,76)
+ylim(0,.5)
#+annotate("text", x = 40, y = .4, label = "Some text")
+geom_vline(xintercept=40)
+geom_vline(xintercept=60)
+theme_classic())

ggplot.save(percplot,'allreg_corrprobno'+'.png',dpi=300)




