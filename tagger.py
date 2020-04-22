import pandas as pd
from tensorflow import keras
import numpy as np
import os.path
from datetime import datetime
from collections import defaultdict
# Needed libraries
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from root_numpy import array2tree, array2root
from dnn_functions import *
from samplesMINIAOD2018 import *

# Configure parameters
pd_folder = 'dataframes/v0_SUSY_calo_MINIAOD_2018/'
tagger_pd_folder = 'dataframes_tagger/v0_SUSY_calo_MINIAOD_2018/'
result_folder = 'model_weights/v0_SUSY_calo_MINIAOD_2018/'
tagger_result_folder = 'model_weights_tagger/v0_SUSY_calo_MINIAOD_2018/'

sgn = ['VBFH_M15_ctau100','VBFH_M20_ctau100','VBFH_M25_ctau100','VBFH_M15_ctau500','VBFH_M20_ctau500','VBFH_M25_ctau500','VBFH_M15_ctau1000','VBFH_M20_ctau1000','VBFH_M25_ctau1000','VBFH_M15_ctau2000','VBFH_M20_ctau2000','VBFH_M25_ctau2000','VBFH_M15_ctau5000','VBFH_M20_ctau5000','VBFH_M25_ctau5000','VBFH_M15_ctau10000','VBFH_M20_ctau10000','VBFH_M25_ctau10000']
bkg = ['ZJetsToNuNu','DYJetsToLL','WJetsToLNu','QCD','VV','TTbar','ST','DYJetsToQQ','WJetsToQQ']
#Test!!!
sgn = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000']

bkg = ['ZJetsToNuNu','WJetsToLNu','QCD','TTbar','VV']
#bkg = ['WJetsToLNu','QCD','TTbar','VV']
#bkg = ['WJetsToLNu']
#bkg = ['QCD']
sgn = []
sgn = ['ggH_MH1000_MS400_ctau500','ggH_MH1000_MS400_ctau1000','ggH_MH1000_MS400_ctau2000','ggH_MH1000_MS400_ctau5000','ggH_MH1000_MS400_ctau10000', 'ggH_MH1000_MS150_ctau500','ggH_MH1000_MS150_ctau1000','ggH_MH1000_MS150_ctau2000','ggH_MH1000_MS150_ctau5000','ggH_MH1000_MS150_ctau10000']
bkg = []
#sgn = ['SUSY_mh400_pl1000']#,'SUSY_mh300_pl1000']
#sgn = ['VBFH_M15_ctau1000','VBFH_M20_ctau1000','VBFH_M25_ctau1000']

train_percentage = 0.8

##Define features
#cols =     ['HT','MEt_pt','MEt_phi','MEt_sign','MinJetMetDPhi','nCHSJets','nElectrons','nMuons','nPhotons','nTaus','j0_pt','j1_pt','j0_nTrackConstituents','j1_nTrackConstituents','j0_nConstituents','j1_nConstituents','j0_nSelectedTracks','j1_nSelectedTracks','j0_nTracks3PixelHits','j1_nTracks3PixelHits','j0_nHadEFrac','j1_nHadEFrac','j0_cHadEFrac','j1_cHadEFrac']

# define your variables here
var_list = [#'EventNumber','RunNumber','LumiNumber','EventWeight','isMC',#not to be trained on!
#'isVBF',
#'HT','MEt_pt',
#'MEt_phi',
#'MEt_sign','MinJetMetDPhi',
#'nCHSJets','nElectrons','nMuons','nPhotons','nTaus',#'nPFCandidates','nPFCandidatesTrack'
]
event_list = ['nCHSJets','HT','MEt_pt','MEt_sign','MinJetMetDPhi','nPFCandidates','nPFCandidatesTrack','EventNumber','RunNumber','LumiNumber','EventWeight','is_signal','c_nEvents']

#jets variables
#nj = 5

#Test!!!
nj=6
#model 0:
# -----------
#'nTrackConstituents','nSelectedTracks','nHadEFrac','cHadEFrac','ecalE','hcalE'
#All bkg, one signal (m 400)
#AUC: 0.9487526172376308
# -----------
#'nTrackConstituents','nSelectedTracks','nHadEFrac','cHadEFrac','ecalE','hcalE','muEFrac','eleEFrac','photonEFrac',
#All bkg, one signal (m 400)
#AUC: 0.9321243621893405
# -----------
#'nTrackConstituents','nSelectedTracks','nHadEFrac','cHadEFrac','ecalE','hcalE'
#All bkg, one signal (m 400), upsampling factor 10
#AUC: 

jtype = ['Jets']
jgen = ['isGenMatched']
jnottrain = [
'pt','eta','phi','mass',
]
jfeatures = [
#'pt','eta','phi','mass',
'nTrackConstituents','nSelectedTracks','nHadEFrac', 'cHadEFrac','ecalE','hcalE',
#'pfXWP1000',
'muEFrac','eleEFrac','photonEFrac',
'eleMulti','muMulti','photonMulti','cHadMulti','nHadMulti',
#new
'nHitsMedian','nPixelHitsMedian',
#Causing issues at array2root!......
#'flightDist2d',#0.9537331313349441 10 epochs
#'flightDist3d',#0.9549956701015507 10 epochs
'dRSVJet',#0.9472248275899331 10 epochs
'nVertexTracks',
'CSV',#saturday 
##'nSV',this has a bad offset, avoid
#'sigIP2DMedian',#0.9538348482298966  10 epochs
'SV_mass',
#'theta2DMedian',# nope, fails 10 epochs
##'alphaMax',#! alpha max is causing problems
]
jvar = jgen+jfeatures+jnottrain
jet_list = []
jet_features_list = []

for n in range(nj):
    for t in jtype:
        for v in jvar:
            jet_list.append(str(t)+str(n)+"_"+v)


for f in jfeatures:
    jet_features_list.append("Jet_"+f)

cols = jet_features_list
print("\n")
print(cols)
print(len(cols)," training features!")
print("\n")

#exit()

##Time stamp for saving model
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d%b%Y_%H_%M_%S")
print("Time:", timestampStr)
print("\n")


def convert_dataset(folder,tagger_folder,sgn,bkg):
    print("  Transform per-event into per-jet dataframes...")
    print("\n")

    signal_list = []
    background_list = []
    for a in sgn:
        signal_list += samples[a]['files']

    for b in bkg:
        background_list += samples[b]['files']

    print(signal_list)
    print(background_list)


    ##Prepare train/test sample for signal
    df_pre_train_s = defaultdict()
    df_pre_test_s = defaultdict()
    for n, s in enumerate(signal_list):
        print("   ",n, s)
        ##load train tables
        store_pre_train_s = pd.HDFStore(folder+s+"_train.h5")
        df_pre_train_s[s] = store_pre_train_s.select("df")
        store_pre_test_s = pd.HDFStore(folder+s+"_test.h5")
        df_pre_test_s[s] = store_pre_test_s.select("df")

    df_temp_train_s = defaultdict()
    df_temp_test_s = defaultdict()
    df_conc_train_s = defaultdict()
    df_conc_test_s = defaultdict()
    df_train_s = defaultdict()
    df_test_s = defaultdict()
    #Loop on signals
    for n, s in enumerate(signal_list):
        #Transform per-event into per-jet
        for j in range(nj):
            temp_list = []
            for l in jet_list:
                if str(j) in l:
                    #print(l)
                    temp_list.append(l)
            ##Temp train
            df_temp_train_s[s] = df_pre_train_s[s][temp_list+event_list]
            df_temp_train_s[s]["Jet_index"] = np.ones(df_temp_train_s[s].shape[0])*j
            ##print("Before renaming")
            ##print(df_temp_train_s)
            ##Temp test
            df_temp_test_s[s] = df_pre_test_s[s][temp_list+event_list]
            df_temp_test_s[s]["Jet_index"] = np.ones(df_temp_test_s[s].shape[0])*j

            #Rename columns
            for v in jvar:
                df_temp_train_s[s].rename(columns={str(t)+str(j)+"_"+v: "Jet_"+str(v)},inplace=True)
                df_temp_test_s[s].rename(columns={str(t)+str(j)+"_"+v: "Jet_"+str(v)},inplace=True)
            #print(df_temp_train_s)

            #Concatenate jets
            if j==0:
                df_conc_train_s[s] = df_temp_train_s[s]
                df_conc_test_s[s] = df_temp_test_s[s]
            else:
                df_conc_train_s[s] = pd.concat([df_conc_train_s[s],df_temp_train_s[s]])
                df_conc_test_s[s] = pd.concat([df_conc_test_s[s],df_temp_test_s[s]])

        ##df_train_s[s] = df_conc_train_s[s][ df_conc_train_s[s]["Jet_isGenMatched"]==1 ]
        ##df_test_s[s] = df_conc_test_s[s][ df_conc_test_s[s]["Jet_isGenMatched"]==1 ]
        ##no selections at the moment
        df_train_s[s] = df_conc_train_s[s]
        df_test_s[s] = df_conc_test_s[s]
        print(s, df_train_s[s])
        ##write h5
        df_train_s[s].to_hdf(tagger_folder+'/'+s+'_train.h5', 'df', format='table')
        print("  "+tagger_folder+"/"+s+"_train.h5 stored")
        df_test_s[s].to_hdf(tagger_folder+'/'+s+'_test.h5', 'df', format='table')
        print("  "+tagger_folder+"/"+s+"_test.h5 stored")
        print("  -------------------   ")


    ##Prepare train/test sample for background
    df_pre_train_b = defaultdict()
    df_pre_test_b = defaultdict()
    for n, b in enumerate(background_list):
        print("   ",n, b)
        #load train tables
        store_pre_train_b = pd.HDFStore(folder+b+"_train.h5")
        df_pre_train_b[b] = store_pre_train_b.select("df")
        store_pre_test_b = pd.HDFStore(folder+b+"_test.h5")
        df_pre_test_b[b] = store_pre_test_b.select("df")

    df_temp_train_b = defaultdict()
    df_temp_test_b = defaultdict()
    df_conc_train_b = defaultdict()
    df_conc_test_b = defaultdict()
    df_train_b = defaultdict()
    df_test_b = defaultdict()
    #Loop on signals
    for n, b in enumerate(background_list):
        #Transform per-event into per-jet
        for j in range(nj):
            temp_list = []
            for l in jet_list:
                if str(j) in l:
                    #print(l)
                    temp_list.append(l)
            #Temp train
            df_temp_train_b[b] = df_pre_train_b[b][temp_list+event_list]
            df_temp_train_b[b]["Jet_index"] = np.ones(df_temp_train_b[b].shape[0])*j
            #Temp test
            df_temp_test_b[b] = df_pre_test_b[b][temp_list+event_list]
            df_temp_test_b[b]["Jet_index"] = np.ones(df_temp_test_b[b].shape[0])*j

            #Rename columns
            for v in jvar:
                df_temp_train_b[b].rename(columns={str(t)+str(j)+"_"+v: "Jet_"+str(v)},inplace=True)
                df_temp_test_b[b].rename(columns={str(t)+str(j)+"_"+v: "Jet_"+str(v)},inplace=True)
            #print(df_temp_train_s)

            #Concatenate jets
            if j==0:
                df_conc_train_b[b] = df_temp_train_b[b]
                df_conc_test_b[b] = df_temp_test_b[b]
            else:
                df_conc_train_b[b] = pd.concat([df_conc_train_b[b],df_temp_train_b[b]])
                df_conc_test_b[b] = pd.concat([df_conc_test_b[b],df_temp_test_b[b]])

        #df_train_b[b] = df_conc_train_b[b][ df_conc_train_b[b]["Jet_isGenMatched"]==0 ]
        #df_test_b[b] = df_conc_test_b[b][ df_conc_test_b[b]["Jet_isGenMatched"]==0 ]
        #No selections
        df_train_b[b] = df_conc_train_b[b]
        df_test_b[b] = df_conc_test_b[b]
        print(b, df_train_b[b])
        #write h5
        df_train_b[b].to_hdf(tagger_folder+'/'+b+'_train.h5', 'df', format='table')
        print("  "+tagger_folder+"/"+b+"_train.h5 stored")
        df_test_b[b].to_hdf(tagger_folder+'/'+b+'_test.h5', 'df', format='table')
        print("  "+tagger_folder+"/"+b+"_test.h5 stored")
        print("  -------------------   ")


def prepare_dataset_tagger(folder,sgn,bkg,model_label,weight="EventWeight",upsample_signal_factor=0):
    print("   Preparing input dataset.....   ")
    print("\n")
    if model_label=="":
        model_label=timestampStr

    signal_list = []
    background_list = []
    for a in sgn:
        signal_list += samples[a]['files']

    for b in bkg:
        background_list += samples[b]['files']

    print(signal_list)
    print(background_list)
    
    ##Prepare train/test sample for signal
    for n, s in enumerate(signal_list):
        print("   ",n, s)
        #load train tables
        store_temp_train_s = pd.HDFStore(folder+s+"_train.h5")
        df_temp_train_s = store_temp_train_s.select("df")
        #load test tables
        store_temp_test_s = pd.HDFStore(folder+s+"_test.h5")
        df_temp_test_s = store_temp_test_s.select("df")
        if n==0:
            df_pre_train_s = df_temp_train_s
            df_pre_test_s = df_temp_test_s
        else:
            df_pre_train_s = pd.concat([df_pre_train_s,df_temp_train_s])
            df_pre_test_s = pd.concat([df_pre_test_s,df_temp_test_s])
        #print(df_pre_train_s[jet_list])

    #Transform per-event into per-jet
    for j in range(nj):
        temp_list = []
        for l in jet_list:
            if str(j) in l:
                #print(l)
                temp_list.append(l)
        #Temp train
        df_temp_train_s = df_pre_train_s[temp_list+event_list]
        df_temp_train_s["Jet_index"] = np.ones(df_temp_train_s.shape[0])*j
        #print("Before renaming")
        #print(df_temp_train_s)
        #Temp test
        df_temp_test_s = df_pre_test_s[temp_list+event_list]
        df_temp_test_s["Jet_index"] = np.ones(df_temp_test_s.shape[0])*j

        #Rename columns
        for v in jvar:
            df_temp_train_s.rename(columns={str(t)+str(j)+"_"+v: "Jet_"+str(v)},inplace=True)
            df_temp_test_s.rename(columns={str(t)+str(j)+"_"+v: "Jet_"+str(v)},inplace=True)
        #print(df_temp_train_s)

        #Concatenate jets
        if j==0:
            df_conc_train_s = df_temp_train_s
            df_conc_test_s = df_temp_test_s
        else:
            df_conc_train_s = pd.concat([df_conc_train_s,df_temp_train_s])
            df_conc_test_s = pd.concat([df_conc_test_s,df_temp_test_s])

    df_train_s = df_conc_train_s[ df_conc_train_s["Jet_isGenMatched"]==1 ]
    df_test_s = df_conc_test_s[ df_conc_test_s["Jet_isGenMatched"]==1 ]
    #print("Final:")
    #print(df_train_s)         

    if upsample_signal_factor>0:
        print("   df_train_s.shape[0] before upsampling", df_train_s.shape[0])
        print("   Will enlarge training sample by factor ", upsample_signal_factor)
        df_train_s = pd.concat([df_train_s]*upsample_signal_factor)
        print("   df_train_s.shape[0] AFTER upsampling", df_train_s.shape[0])


    ##Remove negative weights for training!
    print("----Signal training shape before removing negative weights: ")
    print("   df_train_s.shape[0]", df_train_s.shape[0])
    df_train_s = df_train_s[df_train_s['EventWeight']>=0]
    #df_test_s  = df_test_s[df_test_s['EventWeight']>=0]
    print("----Signal training shape after removing negative weights: ")
    print("   df_train_s.shape[0]", df_train_s.shape[0])

    print(df_train_s)
    ##Normalize train weights
    print("   df_train_s.shape[0]", df_train_s.shape[0])
    norm_train_s = df_train_s['EventWeight'].sum(axis=0)
    print("   renorm signal train: ", norm_train_s)
    df_train_s['EventWeightNormalized'] = df_train_s['EventWeight'].div(norm_train_s)
    df_train_s = df_train_s.sample(frac=1).reset_index(drop=True)#shuffle signals

    ##Normalize test weights
    print("   df_test_s.shape[0]", df_test_s.shape[0])
     #only non negative weights used to normalize test sample
    norm_test_s = df_test_s[ df_test_s['EventWeight']>0 ]['EventWeight'].sum(axis=0)
    print("   renorm signal test: ", norm_test_s)
    df_test_s['EventWeightNormalized'] = df_test_s['EventWeight'].div(norm_test_s)
    df_test_s = df_test_s.sample(frac=1).reset_index(drop=True)#shuffle signals
    print("  -------------------   ")
    print("\n")
    ###n_events_s = int(all_sign.shape[0] * train_percentage)
    ###df_train_s = all_sign.head(n_events_s)
    ###df_test_s = all_sign.tail(all_sign.shape[0] - n_events_s)



    ##Prepare train sample for background
    for n, b in enumerate(background_list):
        print("   ",n, b)
        if not os.path.isfile(folder+b+"_train.h5"):
            print("!!!File ", folder+b+"_train.h5", " does not exist! Continuing")
            continue
        #load train tables
        store_temp_train_b = pd.HDFStore(folder+b+"_train.h5")
        df_temp_train_b = store_temp_train_b.select("df")
        #load test tables
        store_temp_test_b = pd.HDFStore(folder+b+"_test.h5")
        df_temp_test_b = store_temp_test_b.select("df")
        if n==0:
            df_pre_train_b = df_temp_train_b
            df_pre_test_b = df_temp_test_b
        else:
            df_pre_train_b = pd.concat([df_pre_train_b,df_temp_train_b])
            df_pre_test_b = pd.concat([df_pre_test_b,df_temp_test_b])

    #Transform per-event into per-jet
    for j in range(nj):
        temp_list = []
        for l in jet_list:
            if str(j) in l:
                #print(l)
                temp_list.append(l)
        #Temp train
        df_temp_train_b = df_pre_train_b[temp_list+event_list]
        df_temp_train_b["Jet_index"] = np.ones(df_temp_train_b.shape[0])*j
        #print("Before renaming, background")
        #print(df_temp_train_b)
        #Temp test
        df_temp_test_b = df_pre_test_b[temp_list+event_list]
        df_temp_test_b["Jet_index"] = np.ones(df_temp_test_b.shape[0])*j

        #Rename columns
        for v in jvar:
            df_temp_train_b.rename(columns={str(t)+str(j)+"_"+v: "Jet_"+str(v)},inplace=True)
            df_temp_test_b.rename(columns={str(t)+str(j)+"_"+v: "Jet_"+str(v)},inplace=True)
        #print(df_temp_train_s)

        #Concatenate jets
        if j==0:
            df_conc_train_b = df_temp_train_b
            df_conc_test_b = df_temp_test_b
        else:
            df_conc_train_b = pd.concat([df_conc_train_b,df_temp_train_b])
            df_conc_test_b = pd.concat([df_conc_test_b,df_temp_test_b])

    df_train_b = df_conc_train_b[ df_conc_train_b["Jet_isGenMatched"]==0 ]
    df_test_b = df_conc_test_b[ df_conc_test_b["Jet_isGenMatched"]==0 ]
    #print("Final background:")
    #print(df_train_b)

    ##Remove negative weights for training!
    print("----Background training shape before removing negative weights: ")
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    df_train_b = df_train_b[df_train_b['EventWeight']>=0]
    #df_test_b  = df_test_b[df_test_b['EventWeight']>=0]
    print("----Background training shape after removing negative weights: ")
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    
    ##Normalize train weights
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    norm_train_b = df_train_b['EventWeight'].sum(axis=0)
    print("   renorm background train: ", norm_train_b)
    df_train_b['EventWeightNormalized'] = df_train_b['EventWeight'].div(norm_train_b)
    df_train_b = df_train_b.sample(frac=1).reset_index(drop=True)#shuffle signals

    ##Normalize test weights
    #Test sample should keep events with negative weights for the next steps of the analysis, but they should be disregarded for computing the normalized test weights
    print("   df_test_b.shape[0]", df_test_b.shape[0])
    norm_test_b = df_test_b[ df_test_b['EventWeight']>0 ]['EventWeight'].sum(axis=0)#sum only non-negative weights for normalization
    print("   renorm background test: ", norm_test_b)
    df_test_b['EventWeightNormalized'] = df_test_b['EventWeight'].div(norm_test_b)
    df_test_b = df_test_b.sample(frac=1).reset_index(drop=True)#shuffle signals

    print("  -------------------   ")
    
    ###n_events_b = int(all_back.shape[0] * train_percentage)
    ###df_train_b = all_back.head(n_events_b)
    ###df_test_b = all_back.tail(all_back.shape[0] - n_events_b)

    print("   Ratio nB/nS: ", df_train_b.shape[0]/df_train_s.shape[0])
    
    ##Prepare global train and test samples
    df_train = pd.concat([df_train_s,df_train_b])
    df_test = pd.concat([df_test_s,df_test_b])

    ##Reshuffle
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test   = df_test.sample(frac=1).reset_index(drop=True)
    df_train.to_hdf(folder+'train_'+model_label+'.h5', 'df', format='table')
    df_test.to_hdf(folder+'test_'+model_label+'.h5', 'df', format='table')
    print("\n")
    print("   "+folder+"train_"+model_label+".h5 stored")
    print("   "+folder+"test_"+model_label+".h5 stored")   
    print("\n")



def prepare_dataset(folder,sgn,bkg,model_label,weight="EventWeight",upsample_signal_factor=0,signal_match_train=True):
    print("   Preparing input dataset.....   ")
    print("\n")
    if model_label=="":
        model_label=timestampStr

    signal_list = []
    background_list = []
    for a in sgn:
        signal_list += samples[a]['files']

    for b in bkg:
        background_list += samples[b]['files']

    print(signal_list)
    print(background_list)

    ##Prepare train/test sample for signal
    for n, s in enumerate(signal_list):
        print("   ",n, s)
        #load train tables
        store_temp_train_s = pd.HDFStore(folder+s+"_train.h5")
        df_temp_train_s = store_temp_train_s.select("df")
        #load test tables
        store_temp_test_s = pd.HDFStore(folder+s+"_test.h5")
        df_temp_test_s = store_temp_test_s.select("df")
        if n==0:
            df_train_s = df_temp_train_s
            df_test_s = df_temp_test_s
        else:
            df_train_s = pd.concat([df_train_s,df_temp_train_s])
            df_test_s = pd.concat([df_test_s,df_temp_test_s])

    if signal_match_train:
        print("  -------------------   ")
        print("    Training signal only on gen matched jets!!!")
        print(" Size before: ", df_train_s.shape[0])
        df_train_s = df_train_s[ df_train_s["Jet_isGenMatched"] !=0 ]
        print(" Size before: ", df_train_s.shape[0])
        print("  -------------------   ")
    
    if upsample_signal_factor>0:
        print("   df_train_s.shape[0] before upsampling", df_train_s.shape[0])
        print("   Will enlarge training sample by factor ", upsample_signal_factor)
        df_train_s = pd.concat([df_train_s]*upsample_signal_factor)
        print("   df_train_s.shape[0] AFTER upsampling", df_train_s.shape[0])

    ##Remove negative weights for training!
    print("----Signal training shape before removing negative weights: ")
    print("   df_train_s.shape[0]", df_train_s.shape[0])
    df_train_s = df_train_s[df_train_s['EventWeight']>=0]
    #df_test_s  = df_test_s[df_test_s['EventWeight']>=0]
    print("----Signal training shape after removing negative weights: ")
    print("   df_train_s.shape[0]", df_train_s.shape[0])

    print(df_train_s)
    ##Normalize train weights
    print("   df_train_s.shape[0]", df_train_s.shape[0])
    norm_train_s = df_train_s['EventWeight'].sum(axis=0)
    print("   renorm signal train: ", norm_train_s)
    df_train_s['EventWeightNormalized'] = df_train_s['EventWeight'].div(norm_train_s)
    df_train_s = df_train_s.sample(frac=1).reset_index(drop=True)#shuffle signals

    ##Normalize test weights
    print("   df_test_s.shape[0]", df_test_s.shape[0])
    norm_test_s = df_test_s['EventWeight'].sum(axis=0)
    print("   renorm signal test: ", norm_test_s)
    df_test_s['EventWeightNormalized'] = df_test_s['EventWeight'].div(norm_test_s)
    df_test_s = df_test_s.sample(frac=1).reset_index(drop=True)#shuffle signals
    print("  -------------------   ")
    print("\n")
    ###n_events_s = int(all_sign.shape[0] * train_percentage)
    ###df_train_s = all_sign.head(n_events_s)
    ###df_test_s = all_sign.tail(all_sign.shape[0] - n_events_s)

    ##Prepare train sample for background
    for n, b in enumerate(background_list):
        print("   ",n, b)
        if not os.path.isfile(folder+b+"_train.h5"):
            print("!!!File ", folder+b+"_train.h5", " does not exist! Continuing")
            continue
        #load train tables
        store_temp_train_b = pd.HDFStore(folder+b+"_train.h5")
        df_temp_train_b = store_temp_train_b.select("df")
        #load test tables
        store_temp_test_b = pd.HDFStore(folder+b+"_test.h5")
        df_temp_test_b = store_temp_test_b.select("df")
        if n==0:
            df_train_b = df_temp_train_b
            df_test_b = df_temp_test_b
        else:
            df_train_b = pd.concat([df_train_b,df_temp_train_b])
            df_test_b = pd.concat([df_test_b,df_temp_test_b])

    ##Remove negative weights for training!
    print("----Background training shape before removing negative weights: ")
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    df_train_b = df_train_b[df_train_b['EventWeight']>=0]
    #df_test_b  = df_test_b[df_test_b['EventWeight']>=0]
    print("----Background training shape after removing negative weights: ")
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    
    ##Normalize train weights
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    norm_train_b = df_train_b['EventWeight'].sum(axis=0)
    print("   renorm background train: ", norm_train_b)
    df_train_b['EventWeightNormalized'] = df_train_b['EventWeight'].div(norm_train_b)
    df_train_b = df_train_b.sample(frac=1).reset_index(drop=True)#shuffle signals

    ##Normalize test weights
    print("   df_test_b.shape[0]", df_test_b.shape[0])
    norm_test_b = df_test_b['EventWeight'].sum(axis=0)
    print("   renorm background test: ", norm_test_b)
    df_test_b['EventWeightNormalized'] = df_test_b['EventWeight'].div(norm_test_b)
    df_test_b = df_test_b.sample(frac=1).reset_index(drop=True)#shuffle signals

    print("  -------------------   ")
    
    ###n_events_b = int(all_back.shape[0] * train_percentage)
    ###df_train_b = all_back.head(n_events_b)
    ###df_test_b = all_back.tail(all_back.shape[0] - n_events_b)

    print("\n")
    print("   Ratio nB/nS: ", df_train_b.shape[0]/df_train_s.shape[0])
    
    ##Prepare global train and test samples
    df_train = pd.concat([df_train_s,df_train_b])
    df_test = pd.concat([df_test_s,df_test_b])

    ##Reshuffle
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test   = df_test.sample(frac=1).reset_index(drop=True)
    df_train.to_hdf(folder+'train_'+model_label+'.h5', 'df', format='table')
    df_test.to_hdf(folder+'test_'+model_label+'.h5', 'df', format='table')
    print("\n")
    print("   "+folder+"train_"+model_label+".h5 stored")
    print("   "+folder+"test_"+model_label+".h5 stored")   
    print("\n")
    
def fit_model(folder,result_folder,features,is_signal,weight,n_epochs,n_batch_size,patience_val,val_split,model_label,ignore_empty_jets_train=True):
    print("\n")
    print("   Fitting model.....   ")
    print("\n")
    if model_label=="":
        model_label=timestampStr
    ##Define model
    model = keras.models.Sequential()
    #model0 for getting started
    #model.add(keras.layers.Dense(16, input_shape = (len(features),), activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.3))
    #model.add(keras.layers.Dense(2, activation='softmax'))
    #model1
    #model.add(keras.layers.Dense(16, input_shape = (len(features),), activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.3))
    #model.add(keras.layers.Dense(2, activation='softmax'))
    #model.summary()
    #model2
    #model.add(keras.layers.Dense(8, input_shape = (len(features),), activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.3))
    ###model.add(keras.layers.Dense(4, activation='relu'))
    ###model.add(keras.layers.Dropout(rate=0.3))
    #model.add(keras.layers.Dense(2, activation='softmax'))
    #model.summary()
    #model3
    model.add(keras.layers.Dense(24, input_shape = (len(features),), activation='relu'))
    model.add(keras.layers.Dropout(rate=0.3))
    #model.add(keras.layers.Dense(4, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.summary()
    ##model.add(keras.layers.Dense(16, activation='relu'))
    ##model.add(keras.layers.Dropout(rate=0.3))
    
    ##Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics = ["accuracy"])
    #custom_adam:
    #custom_adam = keras.optimizers.Adam(learning_rate=0.001/2., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_adam, metrics = ["accuracy"])

    ##Callbacks
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_val, verbose=0, mode='auto')
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=result_folder+'best_model_'+model_label+'.h5', monitor='val_loss', save_best_only=True)

    ##Read train sample
    store = pd.HDFStore(folder+"train_"+model_label+".h5")
    df_train = store.select("df")
    if ignore_empty_jets_train:
        print("\n")
        print("    Ignore empty jets at training!!!!!!")
        print("\n")
        df_train = df_train[ df_train["Jet_isGenMatched"]!=-1 ]
    print(df_train[is_signal])
    print(df_train[is_signal].sum(axis=0))

    ##Fit model
    #train is 60%, test is 20%, val is 20%
    histObj = model.fit(df_train[features].as_matrix(), df_train[is_signal].as_matrix(), epochs=n_epochs, batch_size=n_batch_size, sample_weight=df_train[weight].as_matrix(), validation_split=val_split, callbacks=[early_stop, checkpoint])
    #validation_data=(df_val[features].as_matrix(), df_val["is_signal"].as_matrix(), df_val["EventWeight"].as_matrix()))#, batch_size=128) 
    histObj.name='model_'+model_label # name added to legend
    plot = plotLearningCurves(histObj)# the above defined function to plot learning curves
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.png')
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.pdf')
    print("Plot saved in: ", result_folder+'loss_accuracy_'+model_label+'.png')
    output_file = 'model_'+model_label
    model.save(result_folder+output_file+'.h5')
    del model
    print("Model saved in ", result_folder+output_file+'.h5')
    #plot.show()

def evaluate_model(folder,result_folder,features,is_signal,weight,n_batch_size,model_label,signal_match_test,ignore_empty_jets_test):
    print("\n")
    print("   Evaluating performances of the model.....   ")
    print("\n")
    if model_label=="":
        model_label=timestampStr
    output_file = 'model_'+model_label
    print("Loading model... ", result_folder+output_file+'.h5')
    model = keras.models.load_model(result_folder+output_file+'.h5')
    model.summary()
    print("Running on test sample. This may take a moment. . .")

    ##Read test sample
    store = pd.HDFStore(folder+"test_"+model_label+".h5")
    df_test = store.select("df")

    print("    Remove negative weights at testing!!!!!!")
    df_test = df_test.loc[df_test['EventWeight']>=0]

    add_string = ""
    if ignore_empty_jets_test:
        print("\n")
        print("    Ignore empty jets at testing!!!!!!")
        print("\n")
        df_test = df_test.loc[df_test["Jet_isGenMatched"]!=-1]
        add_string+="_ignore_empty_jets"

    if signal_match_test:
        print("\n")
        print("    Ignore not matched jets in signal at testing!!!!!!")
        print("\n")
        df_s = df_test.loc[df_test["is_signal"]==1]
        df_b = df_test.loc[df_test["is_signal"]==0]
        df_s = df_s.loc[df_s["Jet_isGenMatched"]==1]
        df_test = pd.concat([df_b,df_s])
        print(df_test.shape[0],df_s.shape[0],df_b.shape[0])
        add_string+="_signal_matched"

    
    probs = model.predict(df_test[features].as_matrix())#predict probability over test sample
    #print("Negative weights?")
    #print(df_test[df_test[weight]<0])
    #df_test = df_test[df_test[weight]>=0]
    #print(df_test)

    AUC = roc_auc_score(df_test[is_signal], probs[:,1],sample_weight=df_test[weight])
    print("Test Area under Curve = {0}".format(AUC))
    #exit()
    df_test["sigprob"] = probs[:,1]

    df_test.to_hdf(result_folder+'test_score_'+model_label+add_string+'.h5', 'df', format='table')
    print("   "+result_folder+"test_score_"+model_label+add_string+".h5 stored")

    back = np.array(df_test["sigprob"].loc[df_test[is_signal]==0].values)
    sign = np.array(df_test["sigprob"].loc[df_test[is_signal]==1].values)
    back_w = np.array(df_test["EventWeightNormalized"].loc[df_test[is_signal]==0].values)
    sign_w = np.array(df_test["EventWeightNormalized"].loc[df_test[is_signal]==1].values)
    #saves the df_test["sigprob"] column when the event is signal or background
    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    #Let's plot an histogram:
    # * y-values: back/sign probabilities
    # * 50 bins
    # * alpha: filling color transparency
    # * density: it should normalize the histograms to unity
    plt.hist(back, 50, weights=back_w, color='blue', edgecolor='blue', lw=2, label='background', alpha=0.3)#, density=True)
    plt.hist(sign, 50, weights=sign_w, color='red', edgecolor='red', lw=2, label='signal', alpha=0.3)#, density=True)

    plt.xlim([0.0, 1.05])
    plt.xlabel('Event probability of being classified as signal')
    plt.legend(loc="upper right")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(result_folder+'probability_'+output_file+add_string+'.png')
    plt.savefig(result_folder+'probability_'+output_file+add_string+'.pdf')
    #plt.show()

    fpr, tpr, _ = roc_curve(df_test[is_signal], df_test["sigprob"], sample_weight=df_test[weight]) #extract true positive rate and false positive rate
    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.plot(fpr, tpr, color='crimson', lw=2, label='ROC curve (area = {0:.4f})'.format(AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'.pdf')
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'.png')
    #plt.show()
    print("   Plots printed in "+result_folder)

    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.plot(fpr, tpr, color='crimson', lw=2, label='ROC curve (area = {0:.4f})'.format(AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.xlim([0.0001, 1.05])
    plt.xscale('log')
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'_logx.pdf')
    plt.savefig(result_folder+'ROC_'+output_file+add_string+'_logx.png')
    #plt.show()

def write_discriminator_output(folder,result_folder,features,is_signal,weight,n_batch_size,model_label,sample_list=[]):
    if model_label=="":
        model_label=timestampStr
    output_file = 'model_'+model_label
    print("Loading model... ", result_folder+output_file+'.h5')
    model = keras.models.load_model(result_folder+output_file+'.h5')
    model.summary()
    print("Running on test sample. This may take a moment. . .")
    
    if sample_list==[]:
        ##Read test sample
        store = pd.HDFStore(folder+"test_"+model_label+".h5")
        df = store.select("df")
        df_valid = df.loc[df["Jet_isGenMatched"]!=-1]
        df_invalid = df.loc[df["Jet_isGenMatched"]==-1]

        probs = model.predict(df_valid[features].as_matrix())#predict probability over test sample
        df_valid["Jet_sigprob"] = probs[:,1]
        df_invalid["Jet_sigprob"] = np.ones(df_invalid.shape[0])*(-1)
        df_test = pd.concat([df_valid,df_invalid])
        df_test.to_hdf(result_folder+'test_score_'+model_label+'.h5', 'df', format='table')
        print("   "+result_folder+"test_score_"+model_label+".h5 stored")


    else:

        full_list = []
        for sl in sample_list:
            full_list += samples[sl]['files']

        for sample in full_list:
            print(" ********************* ")
            print(folder+sample+"_test.h5")
            ##Read test sample
            if not os.path.isfile(folder+sample+"_test.h5"):
                print("!!!File ", folder+sample+"_test.h5", " does not exist! Continuing")
                continue
            store = pd.HDFStore(folder+sample+"_test.h5")
            df = store.select("df")
            df_valid = df.loc[df["Jet_isGenMatched"]!=-1]
            df_invalid = df.loc[df["Jet_isGenMatched"]==-1]

            probs = model.predict(df_valid[features].as_matrix())#predict probability over test sample
            df_valid["Jet_sigprob"] = probs[:,1]
            df_invalid["Jet_sigprob"] = np.ones(df_invalid.shape[0])*(-1)
            df_test = pd.concat([df_valid,df_invalid])
            df_test.to_hdf(result_folder+sample+'_score_'+model_label+'.h5', 'df', format='table')
            print("   "+result_folder+sample+"_score_"+model_label+".h5 stored")


def test_to_root(folder,result_folder,output_root_folder,variables,is_signal,model_label,sample_list=[]):

    if not os.path.isdir(output_root_folder+'/model_'+model_label): os.mkdir(output_root_folder+'/model_'+model_label)

    if sample_list==[]:
        print("   Empty sample list, will use full sample . . .")
        ##Read test sample
        store = pd.HDFStore(result_folder+'test_score_'+model_label+'.h5')
        df_test = store.select("df")

        for n, a in enumerate(variables):
            back = np.array(df_test[a].loc[df_test[is_signal]==0].values, dtype=[(a, np.float64)])
            sign = np.array(df_test[a].loc[df_test[is_signal]==1].values, dtype=[(a, np.float64)])
            print(a," back: ", back)
            print(a," sign: ", sign)
            array2root(back, output_root_folder+'/model_'+model_label+'/test_bkg.root', mode='recreate' if n==0 else 'update')
            array2root(sign, output_root_folder+'/model_'+model_label+'/test_sgn.root', mode='recreate' if n==0 else 'update')
        print("  Signal and background root files written : ", output_root_folder+'/'+model_label+'/test_*.root')

    else:
        full_list = []
        for sl in sample_list:
            full_list += samples[sl]['files']

        for sample in full_list:
            print("   Reading sample: ", sample)
            ##Read test sample
            if not os.path.isfile(folder+sample+"_test.h5"):
                print("!!!File ", folder+sample+"_test.h5", " does not exist! Continuing")
                continue

            store = pd.HDFStore(result_folder+sample+"_score_"+model_label+".h5")
            df_test = store.select("df")

            #smaller for testing
            #df_test = df_test.sample(frac=1).reset_index(drop=True)#shuffle
            #df_test = df_test[0:10]

            #print(df_test)
            df_j = defaultdict()
            #Transform per-event into per-jet
            for j in range(nj):
                #print(j)
                df_j[j] = df_test.loc[ df_test["Jet_index"]==float(j) ]
                #print(df_j[j]["Jet_index"])
                #if df_j[j].shape[0]>0: print(df_j[j])
                #temp_list = []
                for f in jvar:
                    #print("Jet_"+f)
                    #print("Jets"+str(j)+"_"+f)
                    df_j[j].rename(columns={"Jet_"+f: "Jets"+str(j)+"_"+f},inplace=True)
                    #if str(j) in l:
                    #    print("\n")
                    #    #temp_list.append(l)
                df_j[j].rename(columns={"Jet_isGenMatched": "Jets"+str(j)+"_isGenMatched"},inplace=True)
                df_j[j].rename(columns={"Jet_index": "Jets"+str(j)+"_index"},inplace=True)
                df_j[j].rename(columns={"Jet_sigprob": "Jets"+str(j)+"_sigprob"},inplace=True)
                #if df_j[j].shape[0]>0: print(df_j[j])

                if j==0:
                    df = df_j[j]
                else:
                    df_temp = pd.merge(df, df_j[j], on=event_list, how='inner')
                    df = df_temp

            #Here, count how many jets are tagged!
            #Reject events with zero jets
            #print(df)
            df = df[ df['nCHSJets']>0]
            #Define variables to counts nTags
            var_tag_sigprob = []
            var_tag_cHadEFrac = []
            for j in range(nj):
                var_tag_sigprob.append("Jets"+str(j)+"_sigprob")
                var_tag_cHadEFrac.append("Jets"+str(j)+"_cHadEFrac")
            #print(var_tag_sigprob)
            wp_sigprob = [0.5,0.6,0.7,0.8,0.9,0.95]
            wp_cHadEFrac = [0.2,0.1,0.05,0.02]
            for wp in wp_sigprob:
                name = str(wp).replace(".","p")
                df['nTags_sigprob_wp'+name] = df[ df[var_tag_sigprob] > wp ].count(axis=1)
            for wp in wp_cHadEFrac:
                name = str(wp).replace(".","p")
                df['nTags_cHadEFrac_wp'+name] = df[ (df[var_tag_cHadEFrac] < wp) & (df[var_tag_cHadEFrac]>-1) ].count(axis=1)

            #!!!#
            #print("\n")
            #print(df)
            #print(nj)
            #print(len(jfeatures)+3)
            #print(len(event_list))
            #print(list(df.columns))
            #df_0 = df_j[0]
            #df_3 = df_j[3]     
            #mergedStuff = pd.merge(df_0, df_3, on=event_list, how='inner')

            #Here I must compare df_j with the same event number and merge it

            newFile = TFile(output_root_folder+'/model_'+model_label+'/'+sample+'.root', 'recreate')
            newFile.cd()
            for n, a in enumerate(list(df.columns)):
                arr = np.array(df[a].values, dtype=[(a, np.float64)])
                ###print(a, " values: ", arr)
                ###array2root(arr, output_root_folder+'/model_'+model_label+'/'+sample+'.root', mode='update')#mode='recreate' if n==0 else 'update')
                if n==0: skim = array2tree(arr)
                else: array2tree(arr, tree=skim)#mode='recreate' if n==0 else 'update')

            skim.Write()
            ##Recreate c_nEvents histogram
            counter = TH1F("c_nEvents", "Event Counter", 1, 0., 1.)
            counter.Sumw2()
            ##Fill counter histogram with the first entry of c_nEvents
            counter.Fill(0., df["c_nEvents"].values[0])
            ##print("counter bin content: ", counter.GetBinContent(1))
            counter.Write()
            newFile.Close()
            ##counter.Delete()

            
            print("  Root file written : ", output_root_folder+'/model_'+model_label+'/'+sample+'.root')

'''
def test_to_root_prev(folder,result_folder,output_root_folder,variables,is_signal,model_label,sample_list=[]):

    if not os.path.isdir(output_root_folder+'/model_'+model_label): os.mkdir(output_root_folder+'/model_'+model_label)

    if sample_list==[]:
        print("   Empty sample list, will use full sample . . .")
        ##Read test sample
        store = pd.HDFStore(result_folder+'test_score_'+model_label+'.h5')
        df_test = store.select("df")

        for n, a in enumerate(var):
            back = np.array(df_test[a].loc[df_test[is_signal]==0].values, dtype=[(a, np.float64)])
            sign = np.array(df_test[a].loc[df_test[is_signal]==1].values, dtype=[(a, np.float64)])
            print(a," back: ", back)
            print(a," sign: ", sign)
            array2root(back, output_root_folder+'/model_'+model_label+'/test_bkg.root', mode='recreate' if n==0 else 'update')
            array2root(sign, output_root_folder+'/model_'+model_label+'/test_sgn.root', mode='recreate' if n==0 else 'update')
        print("  Signal and background root files written : ", output_root_folder+'/'+model_label+'/test_*.root')

    else:
        full_list = []
        for sl in sample_list:
            full_list += samples[sl]['files']

        for sample in full_list:
            ##Read test sample
            if not os.path.isfile(folder+sample+"_test.h5"):
                print("!!!File ", folder+sample+"_test.h5", " does not exist! Continuing")
                continue

            store = pd.HDFStore(result_folder+sample+"_score_"+model_label+".h5")
            df_test = store.select("df")
            newFile = TFile(output_root_folder+'/model_'+model_label+'/'+sample+'.root', 'recreate')
            newFile.cd()
            for n, a in enumerate(var):
                arr = np.array(df_test[a].values, dtype=[(a, np.float64)])
                #print(a, " values: ", arr)
                #array2root(arr, output_root_folder+'/model_'+model_label+'/'+sample+'.root', mode='update')#mode='recreate' if n==0 else 'update')
                if n==0: skim = array2tree(arr)
                else: array2tree(arr, tree=skim)#mode='recreate' if n==0 else 'update')

            skim.Write()
            ##Recreate c_nEvents histogram
            counter = TH1F("c_nEvents", "Event Counter", 1, 0., 1.)
            counter.Sumw2()
            ##Fill counter histogram with the first entry of c_nEvents
            counter.Fill(0., df_test["c_nEvents"].values[0])
            ##print("counter bin content: ", counter.GetBinContent(1))
            counter.Write()
            newFile.Close()
            #counter.Delete()

            
            print("  Root file written : ", output_root_folder+'/model_'+model_label+'/'+sample+'.root')
'''


##model 0, first attempt:
#convert_dataset(pd_folder,tagger_pd_folder,sgn,bkg)
#prepare_dataset(tagger_pd_folder,sgn,bkg,model_label="1",weight="EventWeightNormalized",upsample_signal_factor=10,signal_match_train=True)#15#23 with third bin
#fit_model(tagger_pd_folder,tagger_result_folder,cols,"Jet_isGenMatched","EventWeightNormalized",n_epochs=50,n_batch_size=2000,patience_val=5,val_split=0.25,model_label="1",ignore_empty_jets_train=True)
# upsample factor 2: 0.9408556577426022
# upsample factor 10: 0.9480164129123527
# upsample factor 10, modified adam lr /2. 0.9433966970451685
# upsample factor 10, 6 jets 0.9471452624943855
# upsample factor 10, alphamax added: dies
# upsample factor 20: 0.9465626268565919
# upsample factor 20 plus kinematics: 0.9496183904468481
# upsample factor 10 plus kinematics: 0.949234104601466
# upsample factor 10 plus kinematics, 25 epochs: 0.9521863606840049 --> saved as model1
# model1b:
# upsample factor 10 no kinematics, 50 epochs: 0.9579508652338852

# CORRECT weights
# upsample factor 10 no kinematics, 50 epochs, correct weights: 0.9466711757557005 
# upsample factor 10 no kinematics but b tag stuff, 50 epochs, correct weights:0.9578379084855911
# upsample factor 10, more taginfo: dRSVJet, nVertexTracks, sigIP2DMedian, SV_mass (no flight distance), 50 epochs, 0.9579585847808232
#without sigIP2DMedian (check offset): 0.9496172653085865
#with CSV: 0.949686331436377
#with CSV + flight: 0.9468950783630051

#model1 again, no flight: 0.9508757565340102
#model2, no flight: 0.9480914819968668
#model3, no flight: 0.948631416808169

#evaluate_model(tagger_pd_folder,tagger_result_folder,cols,"Jet_isGenMatched","EventWeightNormalized",n_batch_size=2000,model_label="1",signal_match_test=True,ignore_empty_jets_test=True)
#evaluate_model(tagger_pd_folder,tagger_result_folder,cols,"Jet_isGenMatched","EventWeightNormalized",n_batch_size=2000,model_label="1",signal_match_test=False,ignore_empty_jets_test=True)
#write_discriminator_output(tagger_pd_folder,tagger_result_folder,cols,"Jet_isGenMatched","EventWeightNormalized",n_batch_size=2000,model_label="1",sample_list=sgn+bkg)
#var = cols + ["EventNumber","RunNumber","LumiNumber","EventWeight","isMC","Jet_isGenMatched","Jet_sigprob","Jet_index"]
output_root_files = "root_files_tagger/v0_SUSY_calo_MINIAOD_2018/"

#var+= ["nDTSegments","nStandAloneMuons","nDisplacedStandAloneMuons"]
test_to_root(tagger_pd_folder,tagger_result_folder,output_root_files,event_list+jvar,"is_signal",model_label="1",sample_list=sgn+bkg)
