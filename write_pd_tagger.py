import ROOT
import root_numpy as rnp
import numpy as np
import pandas as pd
import tables
from ROOT import gROOT, TFile, TTree, TObject, TH1, TH1F, AddressOf, TLorentzVector
from dnn_functions import *
gROOT.ProcessLine('.L Objects.h' )
from ROOT import JetType, CaloJetType, MEtType, CandidateType, DT4DSegmentType, CSCSegmentType#, TrackType

# storage folder of the original root files
folder = '/nfs/dust/cms/group/cms-llp/v0_SUSY_calo_MINIAOD_2018/'
#sgn = ['VBFH_M15_ctau100','VBFH_M20_ctau100','VBFH_M25_ctau100','VBFH_M15_ctau500','VBFH_M20_ctau500','VBFH_M25_ctau500','VBFH_M15_ctau1000','VBFH_M20_ctau1000','VBFH_M25_ctau1000','VBFH_M15_ctau2000','VBFH_M20_ctau2000','VBFH_M25_ctau2000','VBFH_M15_ctau5000','VBFH_M20_ctau5000','VBFH_M25_ctau5000','VBFH_M15_ctau10000','VBFH_M20_ctau10000','VBFH_M25_ctau10000']
bkg = ['ZJetsToNuNu','WJetsToLNu','QCD','VV','TTbar']
sgn = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh250_pl1000','SUSY_mh200_pl1000','SUSY_mh175_pl1000','SUSY_mh150_pl1000','SUSY_mh127_pl1000']
#bkg = ['WJetsToLNu']
bkg = ['QCD','VV','TTbar']
bkg = []
sgn = []
#sgn = ['ggH_MH1000_MS400_ctau500','ggH_MH1000_MS400_ctau1000','ggH_MH1000_MS400_ctau2000','ggH_MH1000_MS400_ctau5000','ggH_MH1000_MS400_ctau10000',
#'ggH_MH1000_MS150_ctau500','ggH_MH1000_MS150_ctau1000','ggH_MH1000_MS150_ctau2000','ggH_MH1000_MS150_ctau5000','ggH_MH1000_MS150_ctau10000']
from samplesMINIAOD2018 import *

#done: 'DYJetsToLL','WJetsToLNu','QCD','VV','TTbar','ST'
#sgn = ['VBFH_M15_ctau1000']
#bkg = []

# define your variables here
var_list = ['EventNumber','RunNumber','LumiNumber','EventWeight','isMC',#not to be trained on!
            'isVBF','HT','MEt.pt','MEt.phi','MEt.sign','MinJetMetDPhi',
            nCHSJets',
            #'nJets',
            'nElectrons','nMuons','nPhotons','nTaus','nPFCandidates','nPFCandidatesTrack']
#jets variables
nj = 6
jtype = ['Jets']
jvar = ['pt','eta','phi','mass','nTrackConstituents','nConstituents','nSelectedTracks','nHadEFrac','cHadEFrac','muEFrac','eleEFrac','photonEFrac','eleMulti','muMulti','photonMulti','cHadMulti','nHadMulti','alphaMax','pfXWP100','pfXWP1000','ecalE','hcalE','sigIP2DMedian','theta2DMedian','POCA_theta2DMedian','nPixelHitsMedian','nHitsMedian','nVertexTracks','CSV','nSV','dRSVJet','flightDist2d','flightDist2dError','flightDist3d','flightDist3dError','nTracksSV','SV_mass']
jvar+=['isGenMatched']
jet_list = []


for n in range(nj):
    for t in jtype:
        for v in jvar:
            jet_list.append(str(t)+"["+str(n)+"]."+v)

print(jet_list)
var_list += jet_list
#,'j0_pt','j1_pt','j0_nTrackConstituents','j1_nTrackConstituents','j0_nConstituents','j1_nConstituents','j0_nSelectedTracks','j1_nSelectedTracks','j0_nTracks3PixelHits','j1_nTracks3PixelHits','j0_nHadEFrac','j1_nHadEFrac','j0_cHadEFrac','j1_cHadEFrac']#,'c_nEvents']#,'is_signal']

variables = []
MEt = MEtType()
#CHSJets = JetType()

def write_h5(folder,output_folder,file_list,test_split,tree_name="",counter_hist="",sel_cut="",obj_sel_cut="",verbose=True):
    print("    Opening ", folder)
    print("\n")
    # loop over files
    for a in file_list:
        print(a)
        for i, ss in enumerate(samples[a]['files']):
            #read number of entries
            oldFile = TFile(folder+ss+'.root', "READ")
            counter = oldFile.Get(counter_hist)#).GetBinContent(1)
            nevents_gen = counter.GetBinContent(1)
            print("  n events gen.: ", nevents_gen)
            oldTree = oldFile.Get(tree_name)
            nevents_tot = oldTree.GetEntries()#?#-1
            tree_weight = oldTree.GetWeight()
            print("   Tree weight:   ",tree_weight)
            #exit()
            oldTree.SetBranchAddress("MEt", AddressOf(MEt, "pt") )
            #oldTree.SetBranchAddress("CHSJets", AddressOf(CHSJets, "pt") );

            #If we apply a cut on the tree, the output will have a different size. Define df later.
            ##initialize data frame
            #df = pd.DataFrame(index = np.arange(nevents_tot) ,columns=var_list)
            #df = df.fillna(0)

            if verbose:
                print("\n")
                #print("   Initialized df for sample: ", file_name)
                print("   Initialized df for sample: ", ss)
                print("   Reading n. events in tree: ", nevents_tot)
                #print("\n")

            if nevents_tot<0:
                print("   Empty tree!!! ")
                continue

            # First loop: check how many events are passing selections
            count = rnp.root2array(folder+ss+'.root', selection = sel_cut, object_selection = obj_sel_cut, treename=tree_name, branches=var_list[0], start=0, stop=nevents_tot)
            nevents=count.shape[0]
            if verbose:
                print("   Cut applied: ", sel_cut)
                print("   Events passing cuts: ", nevents)
                print("\n")            

	    #NEW!!#
            #initialize data frame with the right size
            df = pd.DataFrame(index = np.arange(nevents) ,columns=var_list)
            df = df.fillna(0)
            #NEW!!#  

            # loop over variables
            for nr,variable in enumerate(var_list):
                if "[" in variable:
                    if("eta" in variable or "phi" in variable):
                        default = -9. 
                    elif("SV_mass" in variable or "flightDist2d" in variable or "flightDist3d" in variable or "dRSVJet" in variable or "alphaMax" in variable or "theta2DMedian" in variable):
                        default = -100. 
                    elif("CSV" in variable):
                        default = -99. 
                    elif("sigIP2DMedian" in variable):
                        default = -99999.
                    else:
                        default= -1.
                    b = rnp.root2array(folder+ss+'.root', selection = sel_cut, object_selection = obj_sel_cut, treename=tree_name, branches=(variable,default), start=0, stop=nevents_tot)#we need to run through all the tree!
                    #safe check for vectors of structures: apply hstack
                    b = np.hstack(b)
                else:
                    b = rnp.root2array(folder+ss+'.root', selection = sel_cut, object_selection = obj_sel_cut, treename=tree_name, branches=variable, start=0, stop=nevents_tot)#we need to run through all the tree!
                if '.' in variable:
                    df[variable.replace('.', '_').replace('[', '').replace(']', '')] = b
                    del df[variable]
                elif ('isVBF' in variable or 'isMC' in variable):
                    df[variable] = b.astype(float)
                else:
                    df[variable] = b

            #if needed, rename columns: nJets --> nCHSJets
            #df.rename(columns={"nJets": "nCHSJets"})      
            #add is_signal flag
            df["is_signal"] = np.ones(nevents) if "n3n2" in ss else np.zeros(nevents)
            df["c_nEvents"] = np.ones(nevents) * nevents_gen
            df["EventWeight"] = df["EventWeight"]*tree_weight
            df.rename(columns={"nJets" : "nCHSJets"},inplace=True)
            if verbose:
                print(df)

            #split test and training samples
            #first shuffle
            df.sample(frac=1).reset_index(drop=True)

            #define train and test samples
            n_events = int(df.shape[0] * (1-test_split) )
            df_train = df.head(n_events)
            df_test  = df.tail(df.shape[0] - n_events)
            print("  -------------------   ")
            print("  Events for training: ", df_train.shape[0])
            print("  Events for testing: ", df_test.shape[0])

            #write h5
            df_train.to_hdf(output_folder+'/'+ss+'_train.h5', 'df', format='table')
            print("  "+output_folder+"/"+ss+"_train.h5 stored")
            df_test.to_hdf(output_folder+'/'+ss+'_test.h5', 'df', format='table')
            print("  "+output_folder+"/"+ss+"_test.h5 stored")
            print("  -------------------   ")


'''
def read_h5(folder,file_names):
    for fi, file_name in enumerate(file_names):
        #read hd5
        store = pd.HDFStore(folder+'numpy/'+fileNas[fi]+'.h5')
        df2 = store.select("df")
        print(df2)
'''

print("write")
print("VERY IMPORTANT! selection in rnp.root2array performs an OR of the different conditions!!!!")
write_h5(folder,"dataframes/v0_SUSY_calo_MINIAOD_2018/",sgn+bkg,test_split=0.2,tree_name="ntuple/tree",counter_hist="counter/c_nEvents",sel_cut ="HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v")
##For heavy higgs: 100% test
#write_h5(folder,"dataframes/v0_SUSY_calo_MINIAOD_2018/",sgn+bkg,test_split=1.,tree_name="ntuple/tree",counter_hist="counter/c_nEvents",sel_cut ="HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v")

###write_h5(folder,"dataframes/v0_calo_AOD/",sgn+bkg,test_split=0.2,tree_name="ntuple/tree",counter_hist="counter/c_nEvents",sel_cut ="HT>200",obj_sel_cut="")

#print "read"
#read_h5(folder,file_names)
