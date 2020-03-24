import pandas as pd
from tensorflow import keras
import numpy as np
import os.path
from datetime import datetime
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
result_folder = 'model_weights/v0_SUSY_calo_MINIAOD_2018/'

sgn = ['VBFH_M15_ctau100','VBFH_M20_ctau100','VBFH_M25_ctau100','VBFH_M15_ctau500','VBFH_M20_ctau500','VBFH_M25_ctau500','VBFH_M15_ctau1000','VBFH_M20_ctau1000','VBFH_M25_ctau1000','VBFH_M15_ctau2000','VBFH_M20_ctau2000','VBFH_M25_ctau2000','VBFH_M15_ctau5000','VBFH_M20_ctau5000','VBFH_M25_ctau5000','VBFH_M15_ctau10000','VBFH_M20_ctau10000','VBFH_M25_ctau10000']
bkg = ['ZJetsToNuNu','DYJetsToLL','WJetsToLNu','QCD','VV','TTbar','ST','DYJetsToQQ','WJetsToQQ']
#Test!!!
sgn = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000']

bkg = ['ZJetsToNuNu','WJetsToLNu','QCD','TTbar','VV']
#bkg = ['WJetsToLNu','QCD','TTbar','VV']
#bkg = ['WJetsToLNu']
sgn = ['SUSY_mh400_pl1000']
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
event_list = ['EventNumber','RunNumber','LumiNumber','EventWeight']

#jets variables
#nj = 5

#Test!!!
nj=5
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
jfeatures = [
#'pt','eta','phi','mass',
'nTrackConstituents','nSelectedTracks','nHadEFrac', 'cHadEFrac','ecalE','hcalE',
#'pfXWP1000',
#'muEFrac','eleEFrac','photonEFrac',
##'eleMulti','muMulti','photonMulti','cHadMulti','nHadMulti',
##'alphaMax',#! alpha max is causing problems
]
jvar = jgen+jfeatures
jet_list = []
jet_features_list = []

for n in range(nj):
    for t in jtype:
        for v in jvar:
            jet_list.append(str(t)+str(n)+"_"+v)


for f in jfeatures:
    jet_features_list.append("Jet_"+f)

cols = jet_features_list
print(cols)
print(len(cols)," input features!")

#exit()

##Time stamp for saving model
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d%b%Y_%H_%M_%S")
print("Time:", timestampStr)
print("\n")

def prepare_dataset(folder,sgn,bkg,model_label,weight="EventWeight",upsample_signal_factor=0):
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
    df_test_s  = df_test_s[df_test_s['EventWeight']>=0]
    print("----Signal training shape after removing negative weights: ")
    print("   df_train_s.shape[0]", df_train_s.shape[0])

    print(df_train_s)
    ##Normalize train weights
    print("   df_train_s.shape[0]", df_train_s.shape[0])
    norm_train_s = df_train_s['EventWeight'].sum(axis=0)
    print("   renorm signal train: ", norm_train_s)
    df_train_s['EventWeightNormalized'] = df_train_s['EventWeight'].div(norm_train_s)
    df_train_s.sample(frac=1).reset_index(drop=True)#shuffle signals

    ##Normalize test weights
    print("   df_test_s.shape[0]", df_test_s.shape[0])
    norm_test_s = df_test_s['EventWeight'].sum(axis=0)
    print("   renorm signal test: ", norm_test_s)
    df_test_s['EventWeightNormalized'] = df_test_s['EventWeight'].div(norm_test_s)
    df_test_s.sample(frac=1).reset_index(drop=True)#shuffle signals
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
    df_test_b  = df_test_b[df_test_b['EventWeight']>=0]
    print("----Background training shape after removing negative weights: ")
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    
    ##Normalize train weights
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    norm_train_b = df_train_b['EventWeight'].sum(axis=0)
    print("   renorm background train: ", norm_train_b)
    df_train_b['EventWeightNormalized'] = df_train_b['EventWeight'].div(norm_train_b)
    df_train_b.sample(frac=1).reset_index(drop=True)#shuffle signals

    ##Normalize test weights
    print("   df_test_b.shape[0]", df_test_b.shape[0])
    norm_test_b = df_test_b['EventWeight'].sum(axis=0)
    print("   renorm background test: ", norm_test_b)
    df_test_b['EventWeightNormalized'] = df_test_b['EventWeight'].div(norm_test_b)
    df_test_b.sample(frac=1).reset_index(drop=True)#shuffle signals

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



def prepare_dataset_prev(folder,sgn,bkg,model_label,weight="EventWeight",upsample_signal_factor=0):
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

    
    if upsample_signal_factor>0:
        print("   df_train_s.shape[0] before upsampling", df_train_s.shape[0])
        print("   Will enlarge training sample by factor ", upsample_signal_factor)
        df_train_s = pd.concat([df_train_s]*upsample_signal_factor)
        print("   df_train_s.shape[0] AFTER upsampling", df_train_s.shape[0])

    ##Remove negative weights for training!
    print("----Signal training shape before removing negative weights: ")
    print("   df_train_s.shape[0]", df_train_s.shape[0])
    df_train_s = df_train_s[df_train_s['EventWeight']>=0]
    df_test_s  = df_test_s[df_test_s['EventWeight']>=0]
    print("----Signal training shape after removing negative weights: ")
    print("   df_train_s.shape[0]", df_train_s.shape[0])

    print(df_train_s)
    ##Normalize train weights
    print("   df_train_s.shape[0]", df_train_s.shape[0])
    norm_train_s = df_train_s['EventWeight'].sum(axis=0)
    print("   renorm signal train: ", norm_train_s)
    df_train_s['EventWeightNormalized'] = df_train_s['EventWeight'].div(norm_train_s)
    df_train_s.sample(frac=1).reset_index(drop=True)#shuffle signals

    ##Normalize test weights
    print("   df_test_s.shape[0]", df_test_s.shape[0])
    norm_test_s = df_test_s['EventWeight'].sum(axis=0)
    print("   renorm signal test: ", norm_test_s)
    df_test_s['EventWeightNormalized'] = df_test_s['EventWeight'].div(norm_test_s)
    df_test_s.sample(frac=1).reset_index(drop=True)#shuffle signals
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
    df_test_b  = df_test_b[df_test_b['EventWeight']>=0]
    print("----Background training shape after removing negative weights: ")
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    
    ##Normalize train weights
    print("   df_train_b.shape[0]", df_train_b.shape[0])
    norm_train_b = df_train_b['EventWeight'].sum(axis=0)
    print("   renorm background train: ", norm_train_b)
    df_train_b['EventWeightNormalized'] = df_train_b['EventWeight'].div(norm_train_b)
    df_train_b.sample(frac=1).reset_index(drop=True)#shuffle signals

    ##Normalize test weights
    print("   df_test_b.shape[0]", df_test_b.shape[0])
    norm_test_b = df_test_b['EventWeight'].sum(axis=0)
    print("   renorm background test: ", norm_test_b)
    df_test_b['EventWeightNormalized'] = df_test_b['EventWeight'].div(norm_test_b)
    df_test_b.sample(frac=1).reset_index(drop=True)#shuffle signals

    print("  -------------------   ")
    
    ###n_events_b = int(all_back.shape[0] * train_percentage)
    ###df_train_b = all_back.head(n_events_b)
    ###df_test_b = all_back.tail(all_back.shape[0] - n_events_b)
    
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
    
def fit_model(folder,result_folder,features,is_signal,weight,n_epochs,n_batch_size,patience_val,val_split,model_label):
    print("\n")
    print("   Fitting model.....   ")
    print("\n")
    if model_label=="":
        model_label=timestampStr
    ##Define model
    model = keras.models.Sequential()
    #model0 for getting started
    model.add(keras.layers.Dense(16, input_shape = (len(cols),), activation='relu'))
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.summary()
    
    ##Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics = ["accuracy"])
    #custom_adam:
    #custom_adam = keras.optimizers.Adam(learning_rate=0.001/10., beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_adam, metrics = ["accuracy"])

    ##Callbacks
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_val, verbose=0, mode='auto')
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=result_folder+'best_model_'+model_label+'.h5', monitor='val_loss', save_best_only=True)

    ##Read train sample
    store = pd.HDFStore(folder+"train_"+model_label+".h5")
    df_train = store.select("df")
    print(df_train[is_signal])
    print(df_train[is_signal].sum(axis=0))

    ##Fit model
    #train is 60%, test is 20%, val is 20%
    histObj = model.fit(df_train[features].as_matrix(), df_train[is_signal].as_matrix(), epochs=n_epochs, batch_size=n_batch_size, sample_weight=df_train[weight].as_matrix(), validation_split=val_split, callbacks=[early_stop, checkpoint])
    #validation_data=(df_val[cols].as_matrix(), df_val["is_signal"].as_matrix(), df_val["EventWeight"].as_matrix()))#, batch_size=128) 
    histObj.name='model_'+model_label # name added to legend
    plot = plotLearningCurves(histObj)# the above defined function to plot learning curves
    plot.savefig(result_folder+'loss_accuracy_'+model_label+'.png')
    print("Plot saved in: ", result_folder+'loss_accuracy_'+model_label+'.png')
    output_file = 'model_'+model_label
    model.save(result_folder+output_file+'.h5')
    del model
    print("Model saved in ", result_folder+output_file+'.h5')
    plot.show()

def evaluate_model(folder,result_folder,features,is_signal,weight,n_batch_size,model_label):
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

    probs = model.predict(df_test[features].as_matrix())#predict probability over test sample
    #print("Negative weights?")
    #print(df_test[df_test[weight]<0])
    #df_test = df_test[df_test[weight]>=0]
    #print(df_test)

    AUC = roc_auc_score(df_test[is_signal], probs[:,1],sample_weight=df_test[weight])
    print("Test Area under Curve = {0}".format(AUC))
    df_test["sigprob"] = probs[:,1]

    df_test.to_hdf(result_folder+'test_score_'+model_label+'.h5', 'df', format='table')
    print("   "+result_folder+"test_score_"+model_label+".h5 stored")

    back = np.array(df_test["sigprob"].loc[df_test[is_signal]==0].values)
    sign = np.array(df_test["sigprob"].loc[df_test[is_signal]==1].values)
    back_w = np.array(df_test["EventWeightNormalized"].loc[df_test[is_signal]==0].values)
    sign_w = np.array(df_test["EventWeightNormalized"].loc[df_test[is_signal]==1].values)
    #saves the df_test["sigprob"] column when the event is signal or background
    plt.figure(figsize=(8,5))
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
    plt.grid(True)
    plt.savefig(result_folder+'probability_'+output_file+'.png')
    plt.show()

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
    plt.savefig(result_folder+'ROC_'+output_file+'.png')
    plt.show()
    print("   Plots printed in "+result_folder)

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
        df_test = store.select("df")

        probs = model.predict(df_test[features].as_matrix())#predict probability over test sample
        df_test["sigprob"] = probs[:,1]
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
            df_test = store.select("df")

            probs = model.predict(df_test[features].as_matrix())#predict probability over test sample
            df_test["sigprob"] = probs[:,1]
            df_test.to_hdf(result_folder+sample+'_score_'+model_label+'.h5', 'df', format='table')
            print("   "+result_folder+sample+"_score_"+model_label+".h5 stored")


def test_to_root(folder,result_folder,output_root_folder,variables,is_signal,model_label,sample_list=[]):

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



##model 0, first attempt:
prepare_dataset(pd_folder,sgn,bkg,model_label="0",weight="EventWeightNormalized",upsample_signal_factor=1)#15#23 with third bin
fit_model(pd_folder,result_folder,cols,"Jet_isGenMatched","EventWeightNormalized",n_epochs=10,n_batch_size=2000,patience_val=5,val_split=0.25,model_label="0")

evaluate_model(pd_folder,result_folder,cols,"Jet_isGenMatched","EventWeightNormalized",n_batch_size=200,model_label="0")
#write_discriminator_output(pd_folder,result_folder,cols,"Jet_isGenMatched","EventWeightNormalized",n_batch_size=2000,model_label="0",sample_list=sgn+bkg)
var = cols + ["EventNumber","RunNumber","LumiNumber","EventWeight","isMC","Jet_isGenMatched","sigprob","Jet_index"]
output_root_files = "root_files/v0_SUSY_calo_MINIAOD_2018/"

#var+= ["nDTSegments","nStandAloneMuons","nDisplacedStandAloneMuons"]
#test_to_root(pd_folder,result_folder,output_root_files,var,"is_signal",model_label="2",sample_list=sgn+bkg)

