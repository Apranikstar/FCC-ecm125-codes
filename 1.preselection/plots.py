import os, copy
#import numpy as np

# --------------------------------------------------------------------------------------------------
# We consider two levels in the first step: level 0 (for checking the plots), and level 1 (for the event pre-selection)
# --------------------------------------------------------------------------------------------------


# list of processes
processList = {
    
    #xsecs need to be scaled by 280/989 ...for xsec of ee -> H ...

    # Semileptonic processes
    "wzp6_ee_Henueqq_ecm125": {"fraction":1},
    "wzp6_ee_Hqqenue_ecm125": {"fraction":1},

    "wzp6_ee_Hmunumuqq_ecm125": {"fraction":1},
    "wzp6_ee_Hqqmunumu_ecm125": {"fraction":1},

    "wzp6_ee_Htaunutauqq_ecm125": {"fraction":1},
    "wzp6_ee_Hqqtaunutau_ecm125": {"fraction":1},
    
    "wzp6_ee_enueqq_ecm125":    {"fraction":1, "crossSection": 0.01382,},
    "wzp6_ee_eeqq_ecm125" :     {"fraction":0.1, "crossSection" : 0.5065 },

    "wzp6_ee_munumuqq_ecm125":  {"fraction":1, "crossSection": 0.006711,},
    "wzp6_ee_mumuqq_ecm125":  {"fraction":0.1, "crossSection": 0.006711,},

    "wzp6_ee_taunutauqq_ecm125":{"fraction":1, "crossSection": 0.006761,},
    "wzp6_ee_tautauqq_ecm125":{"fraction":1, "crossSection": 0.006761,},


    # Fully leptonic Processes
    "wzp6_ee_Htautau_ecm125" :  {"fraction":1, "crossSection" : 0.0001011 },
    "wzp6_ee_Hllnunu_ecm125":   {"fraction":1, "crossSection": 3.187e-5,},

    "wzp6_ee_eenunu_ecm125":    {"fraction":1, "crossSection": 0.3364,},
    "wzp6_ee_mumununu_ecm125":  {"fraction":1, "crossSection": 0.2202,},
    "wzp6_ee_tautaununu_ecm125":{"fraction":1, "crossSection": 0.04265,},
    "wzp6_ee_l1l2nunu_ecm125":  {"fraction":1, "crossSection": 0.005799,},
    "wzp6_ee_tautau_ecm125" :   {"fraction":0.1, "crossSection" : 25.939},

    # Fully hadronic Processes
    "wzp6_ee_Hgg_ecm125":       {"fraction":1, "crossSection": 7.384e-5,},
    "wzp6_ee_Hbb_ecm125" :      {"fraction":0.1, "crossSection" : 0.001685},

    "wzp6_ee_qq_ecm125":        {"fraction":0.01, "crossSection": 363.1,},


    
    
}

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
#prodTag     = "FCCee/winter2023/IDEA/"

#Optional: output directory, default is local running directory
outputDir   = "./output/"
# outputDirEos = "/eos/users/r/rjafaris" #helps the output to be visible in CERNbox (does not work!)

# Define the input dir (optional)
#inputDir    = "./localSamples/"
inputDir    = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"
nCPUS       = -1

# additional/costom C++ functions, defined in header files (optional)
#includePaths = ["functions.h"]
includePaths = ["../functions.h"]#, "MELAVar.h", "geo.h"]

## latest particle transformer model, trained on 9M jets in winter2023 samples
model_name = "fccee_flavtagging_edm4hep_wc_v1"

## model files needed for unit testing in CI
url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)

## model files locally stored on /eos
model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
)
local_preproc = "{}/{}.json".format(model_dir, model_name)
local_model = "{}/{}.onnx".format(model_dir, model_name)

## get local file, else download from url
def get_file_path(url, filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        urllib.request.urlretrieve(url, os.path.basename(url))
        return os.path.basename(url)


weaver_preproc = get_file_path(url_preproc, local_preproc)
weaver_model = get_file_path(url_model, local_model)

from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
)

jetFlavourHelper = None
jetClusteringHelper = None

# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis:

 # ______________________________________________________________________________________________________				
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):

 # ______________________________________________________________________________________________________				
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2

    # define some aliases to be used later on
    
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Electron0", "Electron#0.index")
        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Photon0", "Photon#0.index")
        df = df.Alias("Jet2","Jet#2.index")  
 # ______________________________________________________________________________________________________				
    # get all the leptons and photons from the ReconstructedParticles collection

        df = df.Define("electrons_all", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)",)
        df = df.Define("muons_all", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)",)
        df = df.Define("photons_all", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)",)
        df = df.Define("Jets", "FCCAnalyses::ReconstructedParticle::get(Jet2, ReconstructedParticles)",)     
 # ______________________________________________________________________________________________________
    # select leptons and photons with momentum > # GeV (level 0: lep_p > 0 ; level 1: lep_p > 10)

        df = df.Define("electrons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(electrons_all)",)
        df = df.Define("muons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(muons_all)",)
        df = df.Define("photons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(photons_all)",)
 # ______________________________________________________________________________________________________
    # Isolation:
    
        df = df.Define("electrons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(electrons, ReconstructedParticles)",)
        df = df.Define("electrons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.25)(electrons, electrons_iso)",)

        df = df.Define("muons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(muons, ReconstructedParticles)",)
        df = df.Define("muons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.25)(muons, muons_iso)",)
 
        df = df.Define("photons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(photons, ReconstructedParticles)",)
        df = df.Define("photons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.25)(photons, photons_iso)",)

    # electron variables:
        df = df.Define("IsoElectron_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(electrons_sel_iso)",)

        
        # ______________________________________________________________________________________________________
    # Missing variables:
        df = df.Define("MissingE_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET)")
        df = df.Define("Missing_P", "MissingE_4p[0].P()",)
        df = df.Define("Missing_Pt", "MissingE_4p[0].Pt()",) 

# ______________________________________________________________________________________________________
    # Preselection Cuts
        #df = df.Define("LnuM" , "(IsoElectron_4p[0]+MissingE_4p[0]).M()")
 # ______________________________________________________________________________________________________
    # create a new collection of reconstructed particles removing electrons with p>#

        df = df.Define("ReconstructedParticles_nophotons",
                        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons)",)

        df = df.Define("ReconstructedParticlesNoElectrons",
                        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles_nophotons,electrons)",)

        df = df.Define("ReconstructedParticlesNoleptonsNoPhotons",
                        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoElectrons,muons)",)
 # ____________________________________________________________________________________________
 # 2. Jet Clustring using "jetClusteringHelper": (Durham-kt)
 ##########################################    
        global jetClusteringHelper
        global jetFlavourHelper

    # define jet and run clustering parameters
    # name of collections in EDM root files
        collections = {
            "GenParticles": "Particle",
            "PFParticles": "ReconstructedParticles",
            "PFTracks": "EFlowTrack",
            "PFPhotons": "EFlowPhoton",
            "PFNeutralHadrons": "EFlowNeutralHadron",
            "TrackState": "EFlowTrack_1",
            "TrackerHits": "TrackerHits",
            "CalorimeterHits": "CalorimeterHits",
            "dNdx": "EFlowTrack_2",
            "PathLength": "EFlowTrack_L",
            "Bz": "magFieldBz",
        }      

        collections_noleptons_nophotons = copy.deepcopy(collections)
        collections_noleptons_nophotons["PFParticles"] = "ReconstructedParticlesNoleptonsNoPhotons"

        jetClusteringHelper = ExclusiveJetClusteringHelper(collections_noleptons_nophotons["PFParticles"], 2) # for Njet=2
        df = jetClusteringHelper.define(df)
        
        ## define jet flavour tagging parameters

        jetFlavourHelper = JetFlavourHelper(
            collections_noleptons_nophotons,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
        )
        
        ## define observables for tagger
        df = jetFlavourHelper.define(df)

        ## tagger inference
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

 # ___________________________________________________
 # flavor tagging removed from here (check the tutorial if you need it), AND MOVE IT IN OTHER PLACE
 # ___________________________________________________

    ## define jet clustering parameters N = 2
        jetClusteringHelper_N2 = ExclusiveJetClusteringHelper("ReconstructedParticlesNoleptonsNoPhotons", 2, "N2") 

    # jetClusteringHelper_N2 = ExclusiveJetClusteringHelper("ReconstructedParticles_noiso", 2, "N2")
        df = jetClusteringHelper_N2.define(df)

    # # convert jets to LorentzVectors, using jetClusteringHelper (Durham kt is implemented: hidden from here)
        df = df.Define("Jets_p4", "JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets),)
        df = df.Define("Jets_InMa",  "JetConstituentsUtils::InvariantMass(Jets_p4[0], Jets_p4[1])",)
        df = df.Define("ecm125M", "(Jets_p4[0] + Jets_p4[1] + MissingE_4p[0] + IsoElectron_4p[0]).M()")
        df = df.Define("ecm125E", "(Jets_p4[0] + Jets_p4[1] + MissingE_4p[0] + IsoElectron_4p[0]).E()")
        df = df.Define("ecm125P", "(Jets_p4[0] + Jets_p4[1] + MissingE_4p[0] + IsoElectron_4p[0]).P()")
        df = df.Define("ecm125Pt", "(Jets_p4[0] + Jets_p4[1] + MissingE_4p[0] + IsoElectron_4p[0]).Pt()")


        #-------------------------------------------------------------------
        
        #---------------------------------_______________________

        return df
# __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = [
            "Missing_Pt", "Jets_InMa", "ecm125M","ecm125E","ecm125P","ecm125Pt",
        ]

    # outputs jet properties
        #branchList += jetClusteringHelper.outputBranches()

    # outputs jet scores and constituent breakdown [Automatically, create jets variables in the output, but not j1 and j2]
        #branchList += jetFlavourHelper.outputBranches()

        return branchList

