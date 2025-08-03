import os, copy
#import numpy as np

# --------------------------------------------------------------------------------------------------
# We consider two levels in the first step: level 0 (for checking the plots), and level 1 (for the event pre-selection)
# --------------------------------------------------------------------------------------------------

#print("Missing_Pt > 3")
#print("Jets_InMa > 10")
#print("Iso_Electrons_No == 1")
#print(("Jets_InMa < 43")) 
#at the end of the file uncomment these cuts so you get your desired files. I've removed this to plot the initial events



# list of processes
processList = {
    
    #xsecs need to be scaled by 280/989 ...for xsec of ee -> H ...

    # Semileptonic processes
    "wzp6_ee_Hlnuqq_ecm125":    {"fraction":1, "crossSection": 4.58e-5 * (26.248/45.8),},
    "wzp6_ee_Hqqlnu_ecm125":    {"fraction":1, "crossSection": 3.187e-5 * (26.248/31.87),},

    "wzp6_ee_enueqq_ecm125":    {"fraction":1, "crossSection": 0.01382,},
    "wzp6_ee_eeqq_ecm125" :     {"fraction":1, "crossSection" : 0.5065 },

    "wzp6_ee_munumuqq_ecm125":  {"fraction":1, "crossSection": 0.006711,},
    "wzp6_ee_mumuqq_ecm125":  {"fraction":1, "crossSection": 0.006711,},

    "wzp6_ee_taunutauqq_ecm125":{"fraction":1, "crossSection": 0.006761,},


    # Fully leptonic Processes
    "wzp6_ee_Htautau_ecm125" :  {"fraction":1, "crossSection" : 0.0001011 },
    "wzp6_ee_Hllnunu_ecm125":   {"fraction":1, "crossSection": 3.187e-5,},

    "wzp6_ee_eenunu_ecm125":    {"fraction":1, "crossSection": 0.3364,},
    "wzp6_ee_mumununu_ecm125":  {"fraction":1, "crossSection": 0.2202,},
    "wzp6_ee_tautaununu_ecm125":{"fraction":1, "crossSection": 0.04265,},
    "wzp6_ee_l1l2nunu_ecm125":  {"fraction":1, "crossSection": 0.005799,},
    "wzp6_ee_tautau_ecm125" :   {"fraction":1, "crossSection" : 25.939},

    # Fully hadronic Processes
    "wzp6_ee_Hgg_ecm125":       {"fraction":1, "crossSection": 7.384e-5,},
    "wzp6_ee_Hbb_ecm125" :      {"fraction":1, "crossSection" : 0.001685},

    "wzp6_ee_qq_ecm125":        {"fraction":1, "crossSection": 363.1,},


    
    
}

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
#prodTag     = "FCCee/winter2023/IDEA/"

#Optional: output directory, default is local running directory
outputDir   = "/afs/cern.ch/work/h/hfatehi/yukawa/dijetBKG"
# outputDirEos = "/eos/users/r/rjafaris" #helps the output to be visible in CERNbox (does not work!)

# Define the input dir (optional)
#inputDir    = "./localSamples/"
inputDir    = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"

nCPUS       = -1

# additional/costom C++ functions, defined in header files (optional)
#includePaths = ["functions.h"]
includePaths = ["functions.h"]#, "MELAVar.h", "geo.h"]

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
        #df = df.Define("photons", "FCCAnalyses::ReconstructedParticle::sel_pt(1)(photons_all)",) #David's comment
 # ______________________________________________________________________________________________________
    # Isolation:
    
        df = df.Define("electrons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(electrons, ReconstructedParticles)",)
        df = df.Define("electrons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.25)(electrons, electrons_iso)",)

        df = df.Define("muons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(muons, ReconstructedParticles)",)
        df = df.Define("muons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.25)(muons, muons_iso)",)
 
        df = df.Define("photons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(photons, ReconstructedParticles)",)
        df = df.Define("photons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.25)(photons, photons_iso)",)
        #df = df.Define("Iso_Photons_sel",  "FCCAnalyses::ReconstructedParticle::sel_pt(0)(photons_sel_iso)") 
 # ______________________________________________________________________________________________________
    # Create Iso_Leptons collection (considering both mu and elec in an event)

        #df = df.Define("leptons_sel_iso", "FCCAnalyses::ReconstructedParticle::merge(muons_sel_iso, electrons_sel_iso)",)

# ______________________________________________________________________________________________________
    # Kinematic variables :
        df = df.Define("IsoPhotons_4p","FCCAnalyses::ReconstructedParticle::get_tlv(photons_sel_iso)",)
        df = df.Define("Iso_Photon_P", "IsoPhotons_4p[0].P()",)
        df = df.Define("Iso_Photon_Pt", "IsoPhotons_4p[0].Pt()",) 
        df = df.Define("Iso_Photon_Eta", "IsoPhotons_4p[0].Eta()",) 
        df = df.Define("Iso_Photon_Phi", "IsoPhotons_4p[0].Phi()",)
        df = df.Define("Iso_Photon_Rapidity", "IsoPhotons_4p[0].Rapidity()",)
        df = df.Define("Iso_Photon_Theta", "IsoPhotons_4p[0].Theta()",)  
        df = df.Define("Iso_Photon_M", "IsoPhotons_4p[0].M()",)
        df = df.Define("Iso_Photon_Mt", "IsoPhotons_4p[0].Mt()",) 
        df = df.Define("Iso_Photon_E", "IsoPhotons_4p[0].E()")
        df = df.Define("Iso_Photon_Et", "IsoPhotons_4p[0].Et()")
        df = df.Define("Iso_Photon_CosTheta", "IsoPhotons_4p[0].CosTheta()",) 
        df = df.Define("Iso_Photon_CosPhi", "cos(Iso_Photon_Phi)",) 
        df = df.Define("Iso_Photons_No", "photons_sel_iso.size()")
       
        # ______________________________________________________________________________________________________


        df = df.Define("ELEC4p", "FCCAnalyses::ReconstructedParticle::get_tlv(electrons_sel_iso)",)
        #______________________________________________________________________________________________________
    # electron variables:
        df = df.Define("IsoElectron_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(electrons_sel_iso)",)
        df = df.Define("IsoElectron_3p", "IsoElectron_4p[0].Vect()",)
        df = df.Define("Iso_Electron_P", "IsoElectron_4p[0].P()",)
        df = df.Define("Iso_Electron_Pt", "IsoElectron_4p[0].Pt()",) 
        df = df.Define("Iso_Electron_Eta", "IsoElectron_4p[0].Eta()",) 
        df = df.Define("Iso_Electron_Phi", "IsoElectron_4p[0].Phi()",)
        df = df.Define("Iso_Electron_Rapidity", "IsoElectron_4p[0].Rapidity()",)
        df = df.Define("Iso_Electron_Theta", "IsoElectron_4p[0].Theta()",)  
        df = df.Define("Iso_Electron_M", "IsoElectron_4p[0].M()",)
        df = df.Define("Iso_Electron_Mt", "IsoElectron_4p[0].Mt()",) 
        df = df.Define("Iso_Electron_E", "IsoElectron_4p[0].E()")
        df = df.Define("Iso_Electron_Et", "IsoElectron_4p[0].Et()")
        df = df.Define("Iso_Electron_CosTheta", "IsoElectron_4p[0].CosTheta()",) 
        df = df.Define("Iso_Electron_CosPhi", "TMath::Cos(Iso_Electron_Phi)",) 
        df = df.Define("Iso_Electrons_No", "electrons_sel_iso.size()")
        df = df.Define("Iso_Electron_Charge", "FCCAnalyses::ReconstructedParticle::get_charge(electrons_sel_iso)[0]")
        
        # ______________________________________________________________________________________________________
    # Missing variables:
        df = df.Define("MissingE_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET)")
        df = df.Define("Missing_P", "MissingE_4p[0].P()",)
        df = df.Define("Missing_Pt", "MissingE_4p[0].Pt()",) 
        df = df.Define("Missing_Eta", "MissingE_4p[0].Eta()",) 
        df = df.Define("Missing_Phi", "MissingE_4p[0].Phi()",)
        df = df.Define("Missing_Rapidity", "MissingE_4p[0].Rapidity()",)
        df = df.Define("Missing_Theta", "MissingE_4p[0].Theta()",)  
        df = df.Define("Missing_M", "MissingE_4p[0].M()",)
        df = df.Define("Missing_Mt", "MissingE_4p[0].Mt()",) 
        df = df.Define("Missing_E", "MissingE_4p[0].E()")
        df = df.Define("Missing_Et", "MissingE_4p[0].Et()")
        df = df.Define("Missing_CosTheta", "MissingE_4p[0].CosTheta()",) 
        df = df.Define("Missing_CosPhi", "TMath::Cos(Missing_Phi)",)  
# ______________________________________________________________________________________________________
    # Preselection Cuts
        df = df.Define("LnuM" , "(IsoElectron_4p[0]+MissingE_4p[0]).M()")
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

        df = df.Define("d23", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 2))")  # dmerge from 3 to 2
        df = df.Define("d34", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 3))")  # dmerge from 4 to 3

    # # convert jets to LorentzVectors, using jetClusteringHelper (Durham kt is implemented: hidden from here)
        df = df.Define("Jets_p4", "JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets),)
        # Jets charge
        df = df.Define("Jets_charge", "JetConstituentsUtils::get_charge({})".format(jetClusteringHelper.constituents),)
        # Jets constituents
        df = df.Define("get_jet1_constituents", "JetConstituentsUtils::get_jet_constituents({},0)".format(jetClusteringHelper.constituents),)
        df = df.Define("get_jet2_constituents", "JetConstituentsUtils::get_jet_constituents({},1)".format(jetClusteringHelper.constituents),)
        
        #number of Jets constituents
        df = df.Define("n_jet1_constituent", "FCCAnalyses::ReconstructedParticle::get_n(get_jet1_constituents)",)
        df = df.Define("n_jet2_constituent", "FCCAnalyses::ReconstructedParticle::get_n(get_jet2_constituents)",)
        
        #another method
        df = df.Define("Jet_nconst0", "jet_nconst")
        df = df.Define("Jet_nconst1", "jet_nconst[0]")
        df = df.Define("Jet_nconst2", "jet_nconst[1]")
        # MC event primary vertex---------------------------------------------------------
        df = df.Define("MC_PrimaryVertex",  "FCCAnalyses::MCParticle::get_EventPrimaryVertex(21)( Particle )" )
        # Primary vertex TLorentzVector
        df = df.Define("MC_PrimaryVertex_TLorentz", "TLorentzVector(MC_PrimaryVertex.X(), MC_PrimaryVertex.Y(), MC_PrimaryVertex.Z(), 0.0)")
        
        # displacement - method1
        df = df.Define("displacementdz0", "FCCAnalyses::ReconstructedParticle2Track::XPtoPar_dz(ReconstructedParticles,EFlowTrack_1, MC_PrimaryVertex_TLorentz,magFieldBz[0])[0]") 
        df = df.Define("displacementdxy0", "FCCAnalyses::ReconstructedParticle2Track::XPtoPar_dxy(ReconstructedParticles,EFlowTrack_1, MC_PrimaryVertex_TLorentz,magFieldBz[0])[0]")
        
        df = df.Define("displacementdz1", "FCCAnalyses::ReconstructedParticle2Track::XPtoPar_dz(ReconstructedParticles,EFlowTrack_1, MC_PrimaryVertex_TLorentz,magFieldBz[0])[1]") 
        df = df.Define("displacementdxy1", "FCCAnalyses::ReconstructedParticle2Track::XPtoPar_dxy(ReconstructedParticles,EFlowTrack_1, MC_PrimaryVertex_TLorentz,magFieldBz[0])[1]")
        
    # # Variables:
        df = df.Define("Jets_InMa",  "JetConstituentsUtils::InvariantMass(Jets_p4[0], Jets_p4[1])",)
        
        df = df.Define("Jet1_P3", "Jets_p4[0].Vect()")
        df = df.Define("Jet1_P", "Jets_p4[0].P()")
        df = df.Define("Jet1_Pt", "Jets_p4[0].Pt()")
        df = df.Define("Jet1_Eta", "Jets_p4[0].Eta()")
        df = df.Define("Jet1_Rapidity", "Jets_p4[0].Rapidity()")
        df = df.Define("Jet1_Phi", "Jets_p4[0].Phi()")
        df = df.Define("Jet1_M", "Jets_p4[0].M()")
        df = df.Define("Jet1_Mt", "Jets_p4[0].Mt()")
        df = df.Define("Jet1_E", "Jets_p4[0].E()")
        df = df.Define("Jet1_Et", "Jets_p4[0].Et()")
        df = df.Define("Jet1_Theta", "Jets_p4[0].Theta()")
        df = df.Define("Jet1_CosTheta", "Jets_p4[0].CosTheta()")
        df = df.Define("Jet1_CosPhi", "TMath::Cos(Jet1_Phi)")
        
        df = df.Define("Jet2_P3", "Jets_p4[1].Vect()")
        df = df.Define("Jet2_P", "Jets_p4[1].P()")
        df = df.Define("Jet2_Pt", "Jets_p4[1].Pt()")
        df = df.Define("Jet2_Eta", "Jets_p4[1].Eta()")
        df = df.Define("Jet2_Rapidity", "Jets_p4[1].Rapidity()")
        df = df.Define("Jet2_Phi", "Jets_p4[1].Phi()")
        df = df.Define("Jet2_M", "Jets_p4[1].M()")
        df = df.Define("Jet2_Mt", "Jets_p4[1].Mt()")
        df = df.Define("Jet2_E", "Jets_p4[1].E()")
        df = df.Define("Jet2_Et", "Jets_p4[1].Et()")
        df = df.Define("Jet2_Theta", "Jets_p4[1].Theta()")  
        df = df.Define("Jet2_CosTheta", "Jets_p4[1].CosTheta()") 
        df = df.Define("Jet2_CosPhi", "TMath::Cos(Jet2_Phi)")
        
        df = df.Define("Max_JetsPT","TMath::Max(Jets_p4[0].Pt(),Jets_p4[1].Pt())")
        df = df.Define("Min_JetsPT","TMath::Min(Jets_p4[0].Pt(),Jets_p4[1].Pt())")
        df = df.Define("Max_JetsE","TMath::Max(Jets_p4[0].E(),Jets_p4[1].E())")
        df = df.Define("Min_JetsE","TMath::Min(Jets_p4[0].E(),Jets_p4[1].E())")  
        
        df = df.Define("Jet1_charge", "ROOT::VecOps::Sum(Jets_charge[0])")  # Sum for jet 1
        df = df.Define("Jet2_charge", "ROOT::VecOps::Sum(Jets_charge[1])")  # Sum for jet 2
        


        # Preselection cuts
        #df = df.Filter("Missing_Pt > 3")
        #df = df.Filter("Jets_InMa > 10")
        #df = df.Filter("Iso_Electrons_No == 1")
        #df = df.Filter("Jets_InMa < 43")

        #-------------------------------------------------------------------
        
        #---------------------------------_______________________

        return df
# __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = [
            "Missing_Pt" , "Missing_P" , "Jets_InMa", "Jet1_P", "Jet2_P", 


       
  
             
        ]

    # outputs jet properties
        #branchList += jetClusteringHelper.outputBranches()

    # outputs jet scores and constituent breakdown [Automatically, create jets variables in the output, but not j1 and j2]
        #branchList += jetFlavourHelper.outputBranches()

        return branchList
