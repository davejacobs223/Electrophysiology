# Electrophysiology

Some code for neural unit data recordings. Most is made to work with Ripple's output. But there are also some started code for reading in intan data....

For ripping intan data:
intan_process - will get out the voltage trace and do some basic filtering of potential spikes from a thresehold. Works with intans SDK called intan2py

For ripping ripple files:
NOTE – these ripple related codes require a special environment with python 2 that has pyns loaded. See ripple docs for pyns. The ‘environment’ yaml file can be used to create an anaconda environment that sets up the python 2 environment for pyns. 

\textbf{Nev2mat} – reads nev files to get spike forms on each channel based on the snippets. This code assumes the marker codes reported by Ripple need to be divided by 512 but can be adapted depending on how the codes were set up in the experiment  This code will also make rasters for each channel. Most importantly this outputs matlab files for each channel that can be read into Waveclus for sorting. 

Nsx2mat – reads ns2 files to get out the voltage traces (LFPs).  Files are saved as matlab files. 

For prepping or comparing sorting
Mat2py – take Waveclus sorts and plots rasters, ISIs, spike forms over the session. It also makes a pickle file that has all the spike times for the units that is used by the GLM related scripts.

Analysis of spike time data
Zscore glm nest _R- NOTE: you need to have R set up for this with stats and car packages. This code pulls unit spike dictionaries and builds an encoding model. The model is Spike Count = CS|id + US|id + Probability +  trial. Whereby stimid and UStype are nested within CS and US respectively. Then uses ANOVA to get p vals for the predictors.  This will also write out spike matrices and trial condition matrices into a matrices folder. Units are scored as significant for a given bin (250 ms bin, 50 ms step size) if p<.05 and scored as encoders if p<.05 for 5 consecutive samples (one day the threshold should be derived in a more sophisticated manner).

Decoder_cv – this is a code I was fine tuning before funding ran out. Uses scikitlearn to train a naïve bayes classfier on unit data. It will take the matrices spit out by zscore_glm and, for a given region, see if it can use neural spikes to predict US (though the exact Y to be predicted can be changed in lines 52 and 104). Model predictions are checked with 10 fold cross validation. It will also run the same process with shuffled data
