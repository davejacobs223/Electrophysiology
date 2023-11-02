#!/usr/bin/env python2
#
#Code for strip spikes from Ripples NEV files. Works with their own SDK called pyns so pyns must be installed
#NOTE because Ripple doesnt want to update pyns everything must be run in python 2. see the ENVIRONMENT.yaml to get most of the environment set up
#After the environment setup you can install pyns system wide using ripples instructions. (see the trellis documentation)
#
#
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

def rasterplot(dict,name,ev):
	pyplot.clf()
	spikerate=[]
	for atrial in trialdict.keys():
		SR=len(trialdict[atrial])/2.0
		spikerate.append(SR)
	for atrial in range(1,len(nines)):
		ydata=np.array([atrial]*len(trialdict[atrial]))
		xdata=trialdict[atrial]
		pyplot.plot(xdata,ydata,'|', color='black',markersize=2)
		pyplot.axvline(x=0)
	avgrate=sum(spikerate)/len(spikerate)
	plotname=name+'.png'
	pyplot.title(name+" | SpikeRate (Hz): "+str(avgrate))
	pyplot.xlabel('Peri '+'Event '+str(int(ev))+ " time (sec)")
	pyplot.ylabel('Trial')
	pyplot.xlim([-1, 1])
	pyplot.savefig(plotname)


#current durectory is the vin/pyns/pyns for now...
#get the nev files (aka RIpples recording data file)
files=glob('NEVfiles/*.nev')
#del files[5]
#del files[0]
#files=[files[-1]]
concerns=[]

for file in files:
	nsfile = pyns.NSFile(file)
	dirname=file.split('.')[0]

	#type 3 is the segments (neural spike forms)
	segment_entities = [e for e in nsfile.get_entities() if e.entity_type == 3]
	events= [e for e in nsfile.get_entities() if e.entity_type == 1]
	
	if len(events)>0:
		event=events[0]
	else:
		concerns.append([file,'no events found'])
		eoi=options[0]

	#open things to collect data
	sr=1.0/nsfile.get_file_info()[2]

#GET THE EVENTS...
	eventlist=[]
	for item in range(0,event.item_count):
		evtime=event.get_event_data(item)[0]
		evmark=event.get_event_data(item)[1][0]/512
		eventlist.append([evtime,evmark])
	eventlist=np.array(eventlist)
	# specify the event code
	options=np.unique(eventlist[:,1])[0:].tolist()
	print (options)
	#eoi=float(raw_input('Event of interest?'+ '         Available codes:'+str(options)))

	# alternatively if you know the event code and want to run a batch of files you can comment out the sectio nabove, uncomment the line below
	# and put the event number there....
	eoi=9.0
	
	nines=eventlist[np.array(eventlist)[:,1]== eoi]
	start=nines[:,0]-1
	stop=nines[:,0]+1
	peritime=nines[:,0]
	trials=np.linspace(1,len(start),len(start))
	zippedons=np.column_stack((trials,start,stop,peritime))

	if len(nines[:,0])<42:
		concerns.append([file,'under 40 trials'])
		eoi=options[0]	

	if os.path.exists(dirname[9:]):
		os.chdir(dirname[9:])
	else:
		#make new directory
		os.mkdir(dirname[9:])
		os.chdir(dirname[9:])


	#write out event 
	np.savetxt('events.csv',eventlist,delimiter=',',header="Time,Marker",comments="")
	
	nospikes=[]
	for entity in segment_entities:
		if entity.item_count > 0: 
			channel = entity
			trialdict={}
			chaninfo= np.array(channel.get_seg_source_info()[-1].split(',')).tolist()
			chdfname=chaninfo[0]+chaninfo[1].replace(" ", "")
			chdfname=chdfname.split(' ')[1]	
			chdfname='M'+chdfname	


			for a in range(1,len(zippedons)+1):	
				trialdict.update({a:[]})

			idxs=[]
			spikes=[]
			spikesonchan=channel.item_count

			for spike in range(0,spikesonchan):
				startidx=channel.get_time_by_index(spike)
				idxs.append(startidx)
				spikes.append(channel.get_segment_data(spike)[1].tolist())
				for atrial in range(len(zippedons)):
					if startidx >= zippedons[atrial][1] and startidx <= zippedons[atrial][2]:
						trial=atrial+1
						stamp=startidx-zippedons[atrial][3]
						trialdict[trial].append(stamp)

			dictout={'sr':sr,'index':idxs,'spikes':spikes}			
			filename=chdfname+'.mat'
			if os.path.exists('SPIKE_FILES'):
				pass
			else:
				os.mkdir('SPIKE_FILES')
			os.chdir('SPIKE_FILES')
			savemat(filename,dictout)
			os.chdir('../')
			#saved as a matlab file so we can run with waveclus for sorting...
			print chdfname+' saved as mat'

			if os.path.exists('RASTERS'):
				os.chdir('RASTERS')
			else:
				os.mkdir('RASTERS')
				os.chdir('RASTERS')

			rasterplot(trialdict,chdfname,eoi)

			os.chdir('../')


			#when all spikes are collected for a channel
		else:
			chaninfo= np.array(channel.get_seg_source_info()[-1].split(',')).tolist()
			chdfname=chaninfo[0]+chaninfo[1].replace(" ", "")
			chdfname=chdfname.split(' ')[1]	
			chdfname='M'+chdfname	
			nospikes.append(chdfname)

	nospikes=np.array(nospikes)
	np.savetxt('nospikes.csv',nospikes,delimiter=',',fmt='%s')
	os.chdir('../')



