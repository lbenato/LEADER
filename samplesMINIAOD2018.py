#! /usr/bin/env python

#voms-proxy-init -voms cms
#

#Higgs production cross sections: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHXSWG#Production_cross_sections_and_de
#mH = 125 GeV (check if it' better 125.09!)
    #ggH xsec: 48.58
    #VBF xsec: 3.782
    #WH xsec: 1.373
    #ZH xsec: 0.8839

sample = {
    #QCD
    #cross-sections taken from mcm:
    'QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1': {
        'nevents' : 4097049,
        'xsec' : 246300000.0,#pb
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8-v1': {
        'nevents' : 80684349,
        'xsec' : 28060000.0,#pb
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 18722416,
        'xsec' : 1710000.0,#pb
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 17035891,
        'xsec' : 351300,#pb#B2G-17-005
#        'xsec' : 347500,#x-sec DB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 18929951,
        'xsec'  : 31630,#pb#B2G-17-005
#        'xsec'  : 32060,#x-sec DB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 15629253,
        'xsec' : 6802,#pb#B2G-17-005
#        'xsec'  : 6829,#x-sec DB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 4767100,
        'xsec' : 1206,#pb#B2G-17-005
#        'xsec'  : 1207,#x-sec DB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 3970819,
        'xsec' : 120.4,#pb
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1991645,
        'xsec' : 25.25,#pb
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    ##TTbar
    #x-sec taken from mcm - XSDB are different
    'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 300.9,# on mcm ## 365.34, on XSDB #687.1?
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'TTToHadronic_TuneCP5_13TeV-powheg-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 313.9,# on mcm # #377.96 on XSDB,#687.1?
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 72.1,# on mcm # 88.29, #on XSDB ##687.1?
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    #ST
    'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_ext1-v1' : {
        'nevents' : 1,
        'xsec'    : 34.97,#on XSDB ; can't find it on mcm because GEN-SIM can't be found##
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_ext1-v1' : {
        'nevents' : 1,
        'xsec'    : 34.91,#on XSDB ; can't find it on mcm because GEN-SIM can't be found##
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-madgraph-pythia8_ext1-v4' : {
        'nevents' : 1,
        'xsec'    : 10.32*(1.-0.6760),#B2G-17-005: can't find it on mcm because GEN-SIM can't be found; XSDB is empty##
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 80.95,#B2G-17-005: can't find it on mcm because GEN-SIM can't be found; XSDB is empty##
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 136.02,#B2G-17-005: can't find it on mcm because GEN-SIM can't be found; XSDB is empty##
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    #ZJetsToNuNu
    'ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1' : {
        'nevents' : 1,
        'xsec'    : 384.1,
        'matcheff': 1.,
        'kfactor' : 1.37,
    },
    'ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1' : {
        'nevents' : 1,
        'xsec'    : 118.1,
        'matcheff': 1.,
        'kfactor' : 1.52,
    },
    'ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1' : {
        'nevents' : 1,
        'xsec'    : 14.7,
        'matcheff': 1.,
        'kfactor' : 1.37,
    },
    'ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1' : {
        'nevents' : 1,
        'xsec'    : 3.35,
        'matcheff': 1.,
        'kfactor' : 1.04,
    },
    'ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1' : {
        'nevents' : 1,
        'xsec'    : 1.68,
        'matcheff': 1.,
        'kfactor' : 1.14,
    },
    'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1' : {
        'nevents' : 1,
        'xsec'    : 0.316,
        'matcheff': 1.,
        'kfactor' : 0.88,
    },
    'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1' : {
        'nevents' : 1,
        'xsec'    : 0.0072,
        'matcheff': 1.,
        'kfactor' : 0.88,
    },
    #WJetsToLNu
    'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8-v2' : {
        'nevents' : 1,
        'xsec'    : 52850.0,#second entry on XSDB, 50260.0,
        'matcheff': 1.,
        'kfactor' : 0.88,
    },
    #Diboson
    'WW_TuneCP5_13TeV-pythia8-v2' : {
        'nevents' : 1,
        'xsec'    : 75.8,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WZ_TuneCP5_13TeV-pythia8-v3' : {
        'nevents' : 1,
        'xsec'    : 27.6,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'ZZ_TuneCP5_13TeV-pythia8-v2' : {
        'nevents' : 1,
        'xsec'    : 12.14,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },

    
    #SIGNAL
    'SUSY_mh400_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1.,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SUSY_mh300_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1.,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SUSY_mh250_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1.,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SUSY_mh200_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1.,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SUSY_mh175_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1.,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SUSY_mh150_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1.,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SUSY_mh127_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1.,
        'matcheff': 1.,
        'kfactor' : 1.,
    },    
}



samples = {

    #DATA
    'data_obs' : {
        'order' : 0,
        'files' : ['METRun2016B-03Feb2017_ver2-v2'],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Data",
        'weight': 1.,
        'plot': True,
    },

    # Dummy entry for background sum
    'BkgSum' : {
        'order' : 0,
        'files' : [],
        'fillcolor' : 1,
        'fillstyle' : 3003,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Bkg stat.",
        'weight': 1.,
        'plot': True,
    },

    #BACKGROUNDS
    #QCD combined
    'QCD' : {
        'files' : [
            #'QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            #'QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            #'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8-v1', 'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8-v1', 'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8-v1', 'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8-v1', 'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8-v1', 'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1'
        ],
        'fillcolor' : 920,
        'fillstyle' : 1001,
        'linecolor' : 920,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : 'QCD',
        'title' : 'QCD',
        'weight': 1.,
        'plot': True,
    },

    #TTbar
    'TTbar' : {
        'files' : [
            'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8-v1',
            'TTToHadronic_TuneCP5_13TeV-powheg-pythia8-v1',
            'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8-v1'
            ],
        'fillcolor' : 798,
        'fillstyle' : 1001,
        'linecolor' : 798,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "t#bar{t}",
        'weight': 1.,
        'plot': True,
    },
    'TTbarSemiLep' : {
        'files' : [
            'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8-v1',
            #'TTToHadronic_TuneCP5_13TeV-powheg-pythia8-v1',
            #'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8-v1'
            ],
        'fillcolor' : 798,
        'fillstyle' : 1001,
        'linecolor' : 798,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "t#bar{t}",
        'weight': 1.,
        'plot': True,
    },
    'TTbarNu' : {
        'files' : [
            #'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8-v1',
            #'TTToHadronic_TuneCP5_13TeV-powheg-pythia8-v1',
            'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8-v1'
            ],
        'fillcolor' : 798,
        'fillstyle' : 1001,
        'linecolor' : 798,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "t#bar{t}",
        'weight': 1.,
        'plot': True,
    },

    #Single top
    'ST' : {
        'files' : ['ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_ext1-v1','ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_ext1-v1', 'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-madgraph-pythia8_ext1-v4', 'ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8-v1', 'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8-v1',],  
        'fillcolor' : 801,
        'fillstyle' : 1001,
        'linecolor' : 801,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Single t",
        'weight': 1.,
        'plot': True,
    },

    #ZJetsToNuNu
    'ZJetsToNuNu' : {
        'files' : [
            'ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1',
            'ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1',
            'ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1',
            'ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1',
            'ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1',
            'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1',
            'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1',
            ],  
        'fillcolor' : 856,
        'fillstyle' : 1001,
        'linecolor' : 856,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Z jets to NuNu",
        'weight': 1.,
        'plot': True,
    },
    #WJets
    'WJetsToLNu' : { 
        'files' : ['WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8-v2',],
        'fillcolor' : 881,
        'fillstyle' : 1001,
        'linecolor' : 881,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "W jets to LNu",
        'weight': 1.,
        'plot': True,
    },
    #Dibosons
    'VV' : {
        'files' : ['WW_TuneCP5_13TeV-pythia8-v2', 'WZ_TuneCP5_13TeV-pythia8-v3', 'ZZ_TuneCP5_13TeV-pythia8-v2'],  
        'fillcolor' : 602,
        'fillstyle' : 1001,
        'linecolor' : 602,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "WW, WZ, ZZ",
        'weight': 1.,
        'plot': True,
    },
    #SIGNAL
######################################
#
#  JiaJing's AOD
    'SUSY_mh400_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh400_pl1000'],
        'mass' : 400,
        'ctau' : 1000,
        'fillcolor' : 1,
        'fillstyle' : 0,
        'linecolor' : 1,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 400 GeV, c#tau_{0} = 1 mm",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh300_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh300_pl1000'],
        'mass' : 300,
        'ctau' : 1000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 300 GeV, c#tau_{0} = 1 mm",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh250_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh250_pl1000'],
        'mass' : 250,
        'ctau' : 1000,
        'fillcolor' : 826,
        'fillstyle' : 0,
        'linecolor' : 826,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 250 GeV, c#tau_{0} = 1 mm",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh200_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh200_pl1000'],
        'mass' : 200,
        'ctau' : 1000,
        'fillcolor' : 8,
        'fillstyle' : 0,
        'linecolor' : 8,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 200 GeV, c#tau_{0} = 1 mm",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh175_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh175_pl1000'],
        'mass' : 175,
        'ctau' : 1000,
        'fillcolor' : 826,
        'fillstyle' : 0,
        'linecolor' : 826,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 175 GeV, c#tau_{0} = 1 mm",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh150_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh150_pl1000'],
        'mass' : 150,
        'ctau' : 1000,
        'fillcolor' : 826,
        'fillstyle' : 0,
        'linecolor' : 826,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 150 GeV, c#tau_{0} = 1 mm",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh127_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh127_pl1000'],
        'mass' : 127,
        'ctau' : 1000,
        'fillcolor' : 826,
        'fillstyle' : 0,
        'linecolor' : 826,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 127 GeV, c#tau_{0} = 1 mm",
        'weight': 1.,
        'plot': True,
    },
}
