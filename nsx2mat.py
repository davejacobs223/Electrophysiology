#!/usr/bin/env python2

print "hi"

from matplotlib import pyplot
import numpy as np
import pyns
import pandas as pd
from scipy.ndimage import gaussian_filter
import os
from scipy.io import savemat
from glob import glob
import shutil

print "pyns loaded"



#current durectory is the vin/pyns/pyns for now...
#nsfile = pyns.NSFile('F21043_06172021_mesorand_roundtwoaftermoving0004.nev')

files=glob('*.ns2')
#del files[5]
#del files[0]
files=[files[-2]]
concerns=[]
event=pd.read_csv('eventsflick.csv')
markershift=event['Marker'].to_list()
markershift.insert(0,np.nan)
del markershift[-1]
event['marker-1']=markershift
event=event[event['marker-1']==0.0] # 9 or 0

evstamps=event.drop(columns=['marker-1'])
#only for ssvep
nines=evstamps[evstamps['Marker'].isin([11,12,21,22])]
#nines=evstamps[evstamps['Marker'].isin([20., 46., 43., 42., 45., 44., 41., 47.])]


#drop last trial for SSVEP
#nines=nines[0:-1]

for file in files:
	nsfile = pyns.NSFile(file)
	print file
	dirname=file.split('.')[0]

	#type 3 is the segments
	LFP_entities = [e for e in nsfile.get_entities() if e.entity_type == 2]

	
	# if os.path.exists(dirname[9:]):
	# 	os.chdir(dirname[9:])
	# else:
	# 	#make new directory
	# 	os.mkdir(dirname[9:])
	# 	os.chdir(dirname[9:])


	noLFP=[]
	
	

	for entity in LFP_entities:
		if entity.item_count > 0: 
			channel = entity
			chaninfo= 'chan'+channel.get_analog_info()[-1].split(' ')[-1]#this wont worl it will re count
			sfreq=channel.sample_freq
			lower=int(5*sfreq)
			upper=int(5*sfreq)
			lfpname=chaninfo+'_'+str(sfreq)	
			print lfpname
			trialdict={}
			chandata=channel.get_analog_data()

			for trial in range(len(nines)):
				start=nines['Time'].iloc[trial]
				startms=int(start*1000)
				if startms>=5000:
					mini=startms-lower
				else:
					mini=0
				maxi=startms+upper
				trialLFP=chandata[mini:maxi]
				tnum=chaninfo+'_'+str(trial+1)
				trialdict.update({tnum:trialLFP})
				#t = np.arange(0, len(trialLFP), dtype=float)
				#pyplot.plot(t,trialLFP+(trial*20),linewidth=.5)
				
			if os.path.exists('LFP_FILES'):
				pass
			else:
				os.mkdir('LFP_FILES')
			os.chdir('LFP_FILES')

			savemat(chaninfo+'mat',trialdict)
			os.chdir('../')			

			#t = np.arange(0, len(lfpdata), dtype=float)/sfreq

			
			#yloc=yloc+1


	# 		for a in range(1,len(zippedons)+1):	
	# 			trialdict.update({a:[]})

	# 		idxs=[]
	# 		spikes=[]
	# 		spikesonchan=channel.item_count

	# 		for spike in range(0,spikesonchan):
	# 			startidx=channel.get_time_by_index(spike)
	# 			idxs.append(startidx)
	# 			spikes.append(channel.get_segment_data(spike)[1].tolist())
	# 			for atrial in range(len(zippedons)):
	# 				if startidx >= zippedons[atrial][1] and startidx <= zippedons[atrial][2]:
	# 					trial=atrial+1
	# 					stamp=startidx-zippedons[atrial][3]
	# 					trialdict[trial].append(stamp)

	# 		dictout={'sr':sr,'index':idxs,'spikes':spikes}			
	# 		filename=chdfname+'.mat'

	# 		print chdfname+' saved as mat'

	# 		if os.path.exists('RASTERS'):
	# 			os.chdir('RASTERS')
	# 		else:
	# 			os.mkdir('RASTERS')
	# 			os.chdir('RASTERS')

	# 		rasterplot(trialdict,chdfname,eoi)

	# 		os.chdir('../')


	# 		#when all spikes are collected for a channel
	# 	else:
	# 		chaninfo= np.array(channel.get_seg_source_info()[-1].split(',')).tolist()
	# 		chdfname=chaninfo[0]+chaninfo[1].replace(" ", "")
	# 		chdfname=chdfname.split(' ')[1]	
	# 		chdfname='M'+chdfname	
	# 		nospikes.append(chdfname)

	# nospikes=np.array(nospikes)
	# np.savetxt('nospikes.csv',nospikes,delimiter=',',fmt='%s')
	# os.chdir('../')



