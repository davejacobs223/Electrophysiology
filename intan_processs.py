#for intan

from intan2py import *
from scipy.signal import butter, sosfilt, sosfreqz
import pandas as pd
from scipy import signal

def butter_bandpass(lowcut=200, highcut=10000, fs=30000, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut=200, highcut=10000, fs=30000, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

def get_channel_avgs(data,peakminim=50):
	channelcount=len(result[data])
	main=pd.DataFrame()
	negmain=pd.DataFrame()
	alldata=pd.DataFrame()
	for b in range(channelcount):
		trace=result[data][b]
		filttrace=butter_bandpass_filter(trace)
		alldata['ch'+str(b)]=filttrace.tolist()
		locs=signal.find_peaks(filttrace,peakminim,distance=30000)[0]
		neglocs=signal.find_peaks(-filttrace,peakminim,distance=30000)[0]
		for a in range(len(locs)):
			peakar=filttrace[locs[a]-60:locs[a]+60]
			main['ch'+str(b)+str(a)]=['ch'+str(b)]+peakar.tolist()

		for a in range(len(neglocs)):
			peakar=filttrace[neglocs[a]-60:neglocs[a]+60]
			negmain['ch'+str(b)+str(a)]=['ch'+str(b)]+peakar.tolist()
		

	avg=main.transpose().groupby(0).mean()
	negavg=negmain.transpose().groupby(0).mean()

	return (alldata,avg,negavg)


#begin script


filename = 'rec1_210723_121450.rhd'
result, data_present = load_file(filename)

avgsint.to_hdf('tmp.hdf','avgsint', mode='w')



avgs=get_channel_avgs('amplifier_data')[0]
forfile=avgs.transpose().astype('int16')
forfile.to_numpy().tofile('ff.bin')
#avgsint=avgs.astype('int16')

forfile=avgs.transpose().astype('int16')
avgsint.to_csv('ff.csv')

avgsint.to_hdf('ff.bin',key='avgsint')

#avgs[0].T.plot()
#avgs[1].T.plot()








########################


trace=result['amplifier_data'][1]


filttrace=butter_bandpass_filter(trace)

locs=signal.find_peaks(filttrace,height=50,distance=30000)[0]

mainone=[]
for a in range(len(locs)):
	mainone.append(filttrace[locs[a]-150:locs[a]+150])



neglocs=signal.find_peaks(-filttrace,height=50,distance=30000)[0]







main2=[]
for a in range(len(neglocs)):
	main2.append(filttrace[neglocs[a]-150:neglocs[a]+150])