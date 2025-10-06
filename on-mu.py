import os, copy
#import numpy as np

# --------------------------------------------------------------------------------------------------
# We consider two levels in the first step: level 0 (for checking the plots), and level 1 (for the event pre-selection)
# --------------------------------------------------------------------------------------------------

# list of processes
processList = {
    
    #xsecs need to be scaled by 280/989 ...for xsec of ee -> H ...

    # Semileptonic processes
    "wzp6_ee_Henueqq_ecm125":    {"fraction":1},
    "wzp6_ee_Hqqenue_ecm125":    {"fraction":1},

    "wzp6_ee_Hmunumuqq_ecm125":    {"fraction":1},
    "wzp6_ee_Hqqmunumu_ecm125":    {"fraction":1},

    "wzp6_ee_Htaunutauqq_ecm125":    {"fraction":1, },
    "wzp6_ee_Hqqtaunutau_ecm125":    {"fraction":1, },


    "wzp6_ee_taunutauqq_ecm125":{"fraction":1},
    "wzp6_ee_tautauqq_ecm125":{"fraction":1},

    "wzp6_ee_enueqq_ecm125":{"fraction":1},
    "wzp6_ee_eeqq_ecm125":{"fraction":1},

    "wzp6_ee_munumuqq_ecm125":{"fraction":1},
    "wzp6_ee_mumuqq_ecm125":{"fraction":1},


    # # # Fully leptonic Processes
    "wzp6_ee_Htautau_ecm125" :  {"fraction":1 },
    "wzp6_ee_Hllnunu_ecm125":   {"fraction":1 },

    "wzp6_ee_eenunu_ecm125":    {"fraction":1,},
    "wzp6_ee_mumununu_ecm125":  {"fraction":1,},
    "wzp6_ee_tautaununu_ecm125":{"fraction":1, },
    "wzp6_ee_l1l2nunu_ecm125":  {"fraction":1, },
    "wzp6_ee_tautau_ecm125" :   {"fraction":1},

    # # # Fully hadronic Processes
    "wzp6_ee_Hgg_ecm125":       {"fraction":1},
    "wzp6_ee_Hbb_ecm125" :      {"fraction":1},

    "wzp6_ee_qq_ecm125":        {"fraction":1},
    "p8_ee_ZZ_4tau_ecm125":     {"fraction":1},


    
    
}



# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
#prodTag     = "FCCee/winter2023/IDEA/"

#Optional: output directory, default is local running directory
outputDir   = "/eos/user/h/hfatehi/yukawaBDT/on-shell-muon/"
# outputDirEos = "/eos/users/r/rjafaris" #helps the output to be visible in CERNbox (does not work!)

# Define the input dir (optional)
#inputDir    = "./localSamples/"
inputDir    = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"

nCPUS       = -1

# additional/costom C++ functions, defined in header files (optional)
#includePaths = ["functions.h"]
includePaths = ["functions.h", "GEOFunctions.h", "MELAFunctions.h"]

## latest particle transformer model, trained on 9M jets in winter2023 samples
model_name = "fccee_flavtagging_edm4hep_wc"

## model files needed for unit testing in CI
url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)

## model files locally stored on /eos
model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"
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

class RDFanalysis:

    def analysers(df):

 # ___________
        df = df.Alias("Electron0", "Electron#0.index")
        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Photon0", "Photon#0.index")
 # ______________________________________________________________________________________________________				
        df = df.Define("electrons", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)",)
        df = df.Define("muons", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)",)
        df = df.Define("photons", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)",)
 # ______________________________________________________________________________________________________
    # Isolation:
    
        df = df.Define("electrons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.2)(electrons, ReconstructedParticles)",)
        df = df.Define("electrons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.2)(electrons, electrons_iso)",)

        df = df.Define("muons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.2)(muons, ReconstructedParticles)",)
        df = df.Define("muons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.2)(muons, muons_iso)",)
 
        df = df.Define("photons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.2)(photons, ReconstructedParticles)",)
        df = df.Define("photons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.2)(photons, photons_iso)",)

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
    # electron variables:
        df = df.Define("IsoElectron_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(muons_sel_iso)",)
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
        df = df.Define("Iso_Electrons_No", "muons_sel_iso.size()")
        df = df.Define("Iso_Electron_Charge", "FCCAnalyses::ReconstructedParticle::get_charge(muons_sel_iso)[0]")
        
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
        df = df.Filter("Iso_Electrons_No == 1")
        df = df.Filter("Missing_Pt > 3")

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

        df = df.Filter("event_njet > 1")

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
        #df = df.Define("n_jet1_constituent", "FCCAnalyses::ReconstructedParticle::get_n(get_jet1_constituents)",)
        #df = df.Define("n_jet2_constituent", "FCCAnalyses::ReconstructedParticle::get_n(get_jet2_constituents)",)
        
        #another method
        df = df.Define("Jet_nconst0", "jet_nconst")
        df = df.Define("Jet_nconst1", "jet_nconst[0]")
        
        
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
        df = df.Filter("Jets_InMa < 43")






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
        

        #------------------------------------------------------------------------
        df = df.Define("Jets_delR", "Jets_p4[0].DeltaR(Jets_p4[1])")
        df = df.Define("ILjet1_delR", "IsoElectron_4p[0].DeltaR(Jets_p4[0])")
        df = df.Define("ILjet2_delR", "IsoElectron_4p[0].DeltaR(Jets_p4[1])")
        df = df.Define("Max_DelRLJets","TMath::Max(ILjet1_delR,ILjet2_delR)")
        df = df.Define("Min_DelRLJets","TMath::Min(ILjet1_delR,ILjet2_delR)")
        df = df.Define("Jets_delphi", "Jets_p4[0].DeltaPhi(Jets_p4[1])") 
        df = df.Define("ILjet1_delphi", "IsoElectron_4p[0].DeltaPhi(Jets_p4[0])") 
        df = df.Define("ILjet2_delphi", "IsoElectron_4p[0].DeltaPhi(Jets_p4[1])")
        df = df.Define("Max_DelPhiLJets","TMath::Max(ILjet1_delphi,ILjet2_delphi)")
        df = df.Define("Min_DelPhiLJets","TMath::Min(ILjet1_delphi,ILjet2_delphi)")
        df = df.Define("Jets_deleta", "TMath::Abs(Jet1_Eta - Jet2_Eta)") 
        df = df.Define("ILjet1_deleta", "TMath::Abs(Jet1_Eta - Iso_Electron_Eta)") 
        df = df.Define("ILjet2_deleta", "TMath::Abs(Jet2_Eta - Iso_Electron_Eta)") 
        df = df.Define("Max_DelEtaLJets","TMath::Max(ILjet1_deleta,ILjet2_deleta)")
        df = df.Define("Min_DelEtaLJets","TMath::Min(ILjet1_deleta,ILjet2_deleta)")
        df = df.Define("Jets_delrapi", "TMath::Abs(Jet1_Rapidity - Jet2_Rapidity)") 
        df = df.Define("ILjet1_delrapi", "TMath::Abs(Iso_Electron_Rapidity - Jet1_Rapidity)") 
        df = df.Define("ILjet2_delrapi", "TMath::Abs(Iso_Electron_Rapidity - Jet2_Rapidity)") 
        df = df.Define("Max_DelyLJets","TMath::Max(ILjet1_delrapi,ILjet2_delrapi)")
        df = df.Define("Min_DelyLJets","TMath::Min(ILjet1_delrapi,ILjet2_delrapi)")
        df = df.Define("Jets_deltheta", "Jet1_Theta - Jet2_Theta") 
        df = df.Define("Jets_angle", "Jets_p4[0].Angle(Jets_p4[1].Vect())")
        df = df.Define("ILjet1_angle", "IsoElectron_4p[0].Angle(Jets_p4[0].Vect())")
        df = df.Define("ILjet2_angle", "IsoElectron_4p[0].Angle(Jets_p4[1].Vect())")
        df = df.Define("Jets_cosangle", "TMath::Cos(Jets_angle)")
        df = df.Define("ILjet1_cosangle", "TMath::Cos(ILjet1_angle)")
        df = df.Define("ILjet2_cosangle", "TMath::Cos(ILjet2_angle)")
        df = df.Define("Max_CosLJets","TMath::Max(ILjet1_cosangle,ILjet2_cosangle)")
        df = df.Define("Min_CosLJets","TMath::Min(ILjet1_cosangle,ILjet2_cosangle)") 
        df = df.Define("HT", "Iso_Electron_Pt + Jet1_Pt + Jet2_Pt")
        df = df.Define("Higgs_IM", "(Jets_p4[0] + Jets_p4[1] + MissingE_4p[0] + IsoElectron_4p[0]).M()")
        df = df.Define("LJJ_M", "(Jets_p4[0] + Jets_p4[1] + IsoElectron_4p[0]).M()")
        df = df.Define("LJJ_Mt", "(Jets_p4[0] + Jets_p4[1] + IsoElectron_4p[0]).Mt()")
        df = df.Define("LJ1_M", "(Jets_p4[0] + IsoElectron_4p[0]).M()")
        df = df.Define("LJ1_Mt", "(Jets_p4[0] + IsoElectron_4p[0]).Mt()")
        df = df.Define("LJ2_M", "(Jets_p4[1] + IsoElectron_4p[0]).M()")
        df = df.Define("LJ2_Mt", "(Jets_p4[1] + IsoElectron_4p[0]).Mt()")
        df = df.Define("Lnu_M", "(MissingE_4p[0] + IsoElectron_4p[0]).M()")
        df = df.Define("JJ_M", "(Jets_p4[0] + Jets_p4[1]).M()")
        df = df.Define("JJ_Mt", "(Jets_p4[0] + Jets_p4[1]).Mt()")
        df = df.Define("JJ_E", "(Jets_p4[0] + Jets_p4[1]).E()")
        df = df.Define("lj1_PT", "Iso_Electron_Pt + Jet1_Pt")
        df = df.Define("lj2_PT", "Iso_Electron_Pt + Jet2_Pt")
        df = df.Define("jj_PT", "Jet1_Pt + Jet2_Pt")
        df = df.Define("ljj_y", "(Jets_p4[0] + Jets_p4[1] + IsoElectron_4p[0]).Rapidity()")
        df = df.Define("jj_y", "(Jets_p4[0] + Jets_p4[1]).Rapidity()")
        df = df.Define("lj1_y", "(Jets_p4[0] + IsoElectron_4p[0]).Rapidity()")
        df = df.Define("lj2_y", "(Jets_p4[1] + IsoElectron_4p[0]).Rapidity()")
        df = df.Define("ljj_Phi", "(Jets_p4[0] + Jets_p4[1] + IsoElectron_4p[0]).Phi()")
        df = df.Define("jj_Phi", "(Jets_p4[0] + Jets_p4[1]).Phi()")
        df = df.Define("Wl_M", "(IsoElectron_4p[0] + MissingE_4p[0]).M()",)
        df = df.Define("Wl_Theta", "(IsoElectron_4p[0] + MissingE_4p[0]).Theta()",)
        df = df.Define("Shell_M", "TMath::Max((Jets_p4[0]+Jets_p4[1]).M(),Wl_M)")
        df = df.Define("Off_M", "TMath::Min((Jets_p4[0]+Jets_p4[1]).M(),Wl_M)")
        df = df.Define("CosTheta_MaxjjW", "TMath::Cos(TMath::Max((Jets_p4[0]+Jets_p4[1]).Theta(),Wl_Theta))")
        df = df.Define("CosTheta_MinjjW", "TMath::Cos(TMath::Min((Jets_p4[0]+Jets_p4[1]).Theta(),Wl_Theta))")
        df = df.Define("expD", "125.-MissingE_4p[0].E()-IsoElectron_4p[0].E()-Jets_p4[0].E()-Jets_p4[1].E()")
        
        # MELA Variables
        df = df.Define("mela","FCCAnalyses::MELA::MELACalculator::mela(Jets_p4[0],Jets_p4[1],MissingE_4p[0],IsoElectron_4p[0], Iso_Electron_Charge,Jet1_charge,Jet2_charge)")     
        df = df.Define("Phi", " mela.phi")
        df = df.Define("CosPhi", " mela.cosPhi")
        df = df.Define("Phi1", " mela.phi1")
        df = df.Define("CosPhi1", " mela.cosPhi1")
        df = df.Define("PhiStar", " mela.phiStar")
        df = df.Define("CosPhiStar", " mela.cosPhiStar")
        df = df.Define("ThetaStar", " mela.thetaStar")
        df = df.Define("CosThetaStar", "mela.cosThetaStar")
        df = df.Define("Theta1", " mela.theta1")
        df = df.Define("Costheta1", " mela.cosTheta1")
        df = df.Define("Theta2", " mela.theta2")
        df = df.Define("Costheta2", " mela.cosTheta2")
        
        df = df.Define("Planarity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculatePlanarity(Jet1_P3,Jet2_P3,IsoElectron_3p)")
        df = df.Define("APlanarity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAplanarity(Jet1_P3,Jet2_P3,IsoElectron_3p)")
        df = df.Define("Sphericity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateSphericity(Jet1_P3,Jet2_P3,IsoElectron_3p)")
        df = df.Define("ASphericity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAsphericity(Jet1_P3,Jet2_P3,IsoElectron_3p)")
         
        #####---------------------------------------
        df = df.Define("scoreG1", "recojet_isG[0]")
        df = df.Define("scoreG2", "recojet_isG[1]")
        df = df.Define("scoreU1", "recojet_isU[0]")
        df = df.Define("scoreU2", "recojet_isU[1]")
        df = df.Define("scoreS1", "recojet_isS[0]")
        df = df.Define("scoreS2", "recojet_isS[1]")
        df = df.Define("scoreC1", "recojet_isC[0]")
        df = df.Define("scoreC2", "recojet_isC[1]")
        df = df.Define("scoreB1", "recojet_isB[0]")
        df = df.Define("scoreB2", "recojet_isB[1]")
        df = df.Define("scoreT1", "recojet_isTAU[0]")
        df = df.Define("scoreT2", "recojet_isTAU[1]")
        df = df.Define("scoreD1", "recojet_isD[0]")
        df = df.Define("scoreD2", "recojet_isD[1]")
        #---------------------------------------------------------------------
        df = df.Define("scoreSumG", "recojet_isG[0]+recojet_isG[1]")
        df = df.Define("scoreSumU", "recojet_isU[0]+recojet_isU[1]")
        df = df.Define("scoreSumS", "recojet_isS[0]+recojet_isS[1]")
        df = df.Define("scoreSumC", "recojet_isC[0]+recojet_isC[1]")
        df = df.Define("scoreSumB", "recojet_isB[0]+recojet_isB[1]")
        df = df.Define("scoreSumT", "recojet_isTAU[0]+recojet_isTAU[1]")
        df = df.Define("scoreSumD", "recojet_isD[0]+recojet_isD[1]")
        
        #####---------------------------------------------------------------
        df = df.Define("scoreMultiplyG", "recojet_isG[0]*recojet_isG[1]")
        df = df.Define("scoreMultiplyU", "recojet_isU[0]*recojet_isU[1]")
        df = df.Define("scoreMultiplyS", "recojet_isS[0]*recojet_isS[1]")
        df = df.Define("scoreMultiplyC", "recojet_isC[0]*recojet_isC[1]")
        df = df.Define("scoreMultiplyB", "recojet_isB[0]*recojet_isB[1]")
        df = df.Define("scoreMultiplyT", "recojet_isTAU[0]*recojet_isTAU[1]")
        df = df.Define("scoreMultiplyD", "recojet_isD[0]*recojet_isD[1]")
        

        
# __________________________________________________________________________________________

        return df
# __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = [
       
             "Missing_P","Missing_Pt","Missing_Phi","Missing_Eta","Missing_Theta","Missing_Rapidity","Missing_M","Missing_Mt", 
             "Missing_E","Missing_Et","Missing_CosTheta","Missing_CosPhi",
             "Iso_Photon_P","Iso_Photon_Pt","Iso_Photon_Eta","Iso_Photon_Phi","Iso_Photon_Theta","Iso_Photon_Rapidity",
             "Iso_Photon_M","Iso_Photon_Mt","Iso_Photon_E","Iso_Photon_Et","Iso_Photon_CosTheta","Iso_Photon_CosPhi","Iso_Photons_No",
             "Iso_Electron_P","Iso_Electron_Pt","Iso_Electron_Eta","Iso_Electron_Phi","Iso_Electron_Theta","Iso_Electron_Rapidity","Iso_Electron_M", 
             "Iso_Electron_Mt","Iso_Electron_E","Iso_Electron_Et","Iso_Electron_CosTheta","Iso_Electron_CosPhi","Iso_Electrons_No",
             "Jets_InMa","d23","d34", 
             "Jet1_P","Jet1_Pt","Jet1_E","Jet1_Et","Jet1_Eta","Jet1_Rapidity","Jet1_Phi","Jet1_M","Jet1_Mt","Jet1_Theta","Jet1_CosTheta",
             "Jet1_CosPhi",  
             "Jet2_P","Jet2_Pt","Jet2_E","Jet2_Et","Jet2_Eta","Jet2_Rapidity","Jet2_Phi","Jet2_M","Jet2_Mt","Jet2_Theta","Jet2_CosTheta", 
             "Jet2_CosPhi",    
             "Jets_delR","ILjet1_delR","ILjet2_delR","Jets_delphi","ILjet1_delphi","ILjet2_delphi","Jets_deleta","ILjet1_deleta", 
             "ILjet2_deleta","Jets_delrapi","ILjet1_delrapi","ILjet2_delrapi","Jets_deltheta","Jets_angle","ILjet1_angle","ILjet2_angle", 
             "Jets_cosangle","ILjet1_cosangle","ILjet2_cosangle","HT","Higgs_IM",
             "LJJ_M","LJJ_Mt","LJ1_M","LJ1_Mt","LJ2_M","LJ2_Mt","Lnu_M","JJ_M","JJ_Mt","JJ_E","lj1_PT","lj2_PT","jj_PT","ljj_y","jj_y", 
             "lj1_y","lj2_y","ljj_Phi","jj_Phi","Wl_M","Wl_Theta","Shell_M","Off_M","CosTheta_MaxjjW","CosTheta_MinjjW","expD",
             "Max_JetsPT","Min_JetsPT","Max_JetsE","Min_JetsE","Max_DelRLJets","Min_DelRLJets","Max_DelPhiLJets","Min_DelPhiLJets",
             "Max_DelEtaLJets","Min_DelEtaLJets","Max_DelyLJets","Min_DelyLJets","Max_CosLJets","Min_CosLJets",
             "Phi","CosPhi","Phi1","CosPhi1","PhiStar","CosPhiStar","ThetaStar","CosThetaStar","Theta1","Costheta1","Theta2","Costheta2",
             "Planarity","APlanarity","Sphericity","ASphericity",
             "Jet_nconst0","Jet_nconst1","displacementdz0","displacementdxy0","displacementdz1","displacementdxy1", 
             # Individual scores
    "scoreG1", "scoreG2",
    "scoreU1", "scoreU2",
    "scoreS1", "scoreS2",
    "scoreC1", "scoreC2",
    "scoreB1", "scoreB2",
    "scoreT1", "scoreT2",
    "scoreD1", "scoreD2",

    # Summed scores
    "scoreSumG", "scoreSumU", "scoreSumS",
    "scoreSumC", "scoreSumB", "scoreSumT", "scoreSumD",

    # Multiplied scores
    "scoreMultiplyG", "scoreMultiplyU", "scoreMultiplyS",
    "scoreMultiplyC", "scoreMultiplyB", "scoreMultiplyT", "scoreMultiplyD",
             
        ]

    # outputs jet properties
        #branchList += jetClusteringHelper.outputBranches()

    # outputs jet scores and constituent breakdown [Automatically, create jets variables in the output, but not j1 and j2]
        #branchList += jetFlavourHelper.outputBranches()

        return branchList
