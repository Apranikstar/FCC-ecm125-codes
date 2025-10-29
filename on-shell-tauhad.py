import os, copy
#import numpy as np

# --------------------------------------------------------------------------------------------------
# We consider two levels in the first step: level 0 (for checking the plots), and level 1 (for the event pre-selection)
# --------------------------------------------------------------------------------------------------

# list of processes
processList = {
    
    #xsecs need to be scaled by 280/989 ...for xsec of ee -> H ...

    # #Semileptonic processes
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


    # # # # # Fully leptonic Processes
    "wzp6_ee_Htautau_ecm125" :  {"fraction":1 },
    "wzp6_ee_Hllnunu_ecm125":   {"fraction":1 },

    "wzp6_ee_eenunu_ecm125":    {"fraction":1,},
    "wzp6_ee_mumununu_ecm125":  {"fraction":1,},
    "wzp6_ee_tautaununu_ecm125":{"fraction":1, },
    "wzp6_ee_l1l2nunu_ecm125":  {"fraction":1, },
    "wzp6_ee_tautau_ecm125" :   {"fraction":1},

    # # # # Fully hadronic Processes
    "wzp6_ee_Hgg_ecm125":       {"fraction":1},
    "wzp6_ee_Hbb_ecm125" :      {"fraction":1},

    "wzp6_ee_qq_ecm125":        {"fraction":1},
    "p8_ee_ZZ_4tau_ecm125":     {"fraction":1},


    
    
}



outputDir   = "/eos/user/h/hfatehi/yukawaBDT/on-shell-THad/"

inputDir    = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"

nCPUS       = -1

includePaths = ["functions.h", "GEOFunctions.h", "MELAFunctions.h", "SortJets.h"]

model_name = "fccee_flavtagging_edm4hep_wc"

url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)

model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"
)
local_preproc = "{}/{}.json".format(model_dir, model_name)
local_model = "{}/{}.onnx".format(model_dir, model_name)

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
       

        df = df.Define("Iso_Electrons_No", "electrons_sel_iso.size()")
        df = df.Define("Iso_Muons_No", "muons_sel_iso.size()")
        
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
        df = df.Filter("Iso_Electrons_No == 0")
        df = df.Filter("Iso_Muons_No == 0")
        df = df.Filter("Missing_Pt > 3")


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

        #collections_noleptons_nophotons = copy.deepcopy(collections)
        #collections_noleptons_nophotons["PFParticles"] = "ReconstructedParticles"

        jetClusteringHelper = ExclusiveJetClusteringHelper(collections["PFParticles"], 3) # for Njet=2
        df = jetClusteringHelper.define(df)
        
        ## define jet flavour tagging parameters

        jetFlavourHelper = JetFlavourHelper(
            collections,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
        )
        
        ## define observables for tagger
        df = jetFlavourHelper.define(df)

        ## tagger inference
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

        df = df.Filter("event_njet > 2")

    ## define jet clustering parameters N = 2
        jetClusteringHelper_N3 = ExclusiveJetClusteringHelper("ReconstructedParticles", 3, "N3") 

    # jetClusteringHelper_N2 = ExclusiveJetClusteringHelper("ReconstructedParticles_noiso", 2, "N2")
        df = jetClusteringHelper_N3.define(df)

        df = df.Define("d34", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N3, 3))")  # dmerge from 3 to 2
        df = df.Define("d45", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N3, 4))")  # dmerge from 4 to 3

        df = df.Define("SortedJets", f"FCCAnalyses::JetUtils::JetSorter::sort_jets_by_score({jetClusteringHelper.jets}, recojet_isTAU)")

        df = df.Define("Jets_p4", "JetConstituentsUtils::compute_tlv_jets(SortedJets)")

        #df = df.Define("Jets_charge", "JetConstituentsUtils::get_charge(SortedJetConstituents)",)

        df = df.Define("m_jj",  "(Jets_p4[1]+Jets_p4[2]).M()",)
        df = df.Filter("m_jj < 43")
        
        
        # MC event primary vertex---------------------------------------------------------
        df = df.Define("MC_PrimaryVertex",  "FCCAnalyses::MCParticle::get_EventPrimaryVertex(21)( Particle )" )
        # # Primary vertex TLorentzVector
        df = df.Define("MC_PrimaryVertex_TLorentz", "TLorentzVector(MC_PrimaryVertex.X(), MC_PrimaryVertex.Y(), MC_PrimaryVertex.Z(), 0.0)")
        
        # # displacement - method1
        df = df.Define("displacementdz0", "FCCAnalyses::ReconstructedParticle2Track::XPtoPar_dz(ReconstructedParticles,EFlowTrack_1, MC_PrimaryVertex_TLorentz,magFieldBz[0])[0]") 
        df = df.Define("displacementdxy0", "FCCAnalyses::ReconstructedParticle2Track::XPtoPar_dxy(ReconstructedParticles,EFlowTrack_1, MC_PrimaryVertex_TLorentz,magFieldBz[0])[0]")
        
        df = df.Define("displacementdz1", "FCCAnalyses::ReconstructedParticle2Track::XPtoPar_dz(ReconstructedParticles,EFlowTrack_1, MC_PrimaryVertex_TLorentz,magFieldBz[0])[1]") 
        df = df.Define("displacementdxy1", "FCCAnalyses::ReconstructedParticle2Track::XPtoPar_dxy(ReconstructedParticles,EFlowTrack_1, MC_PrimaryVertex_TLorentz,magFieldBz[0])[1]")
        

    # # Variables:




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
        
        df = df.Define("Jet3_P3", "Jets_p4[2].Vect()")
        df = df.Define("Jet3_P", "Jets_p4[2].P()")
        df = df.Define("Jet3_Pt", "Jets_p4[2].Pt()")
        df = df.Define("Jet3_Eta", "Jets_p4[2].Eta()")
        df = df.Define("Jet3_Rapidity", "Jets_p4[2].Rapidity()")
        df = df.Define("Jet3_Phi", "Jets_p4[2].Phi()")
        df = df.Define("Jet3_M", "Jets_p4[2].M()")
        df = df.Define("Jet3_Mt", "Jets_p4[2].Mt()")
        df = df.Define("Jet3_E", "Jets_p4[2].E()")
        df = df.Define("Jet3_Et", "Jets_p4[2].Et()")
        df = df.Define("Jet3_Theta", "Jets_p4[2].Theta()")
        df = df.Define("Jet3_CosTheta", "Jets_p4[2].CosTheta()")
        df = df.Define("Jet3_CosPhi", "TMath::Cos(Jet3_Phi)")

        df = df.Define("Max_JetsPT", 
               "TMath::Max(Jets_p4[0].Pt(), TMath::Max(Jets_p4[1].Pt(), Jets_p4[2].Pt()))")
        df = df.Define("Min_JetsPT", 
               "TMath::Min(Jets_p4[0].Pt(), TMath::Min(Jets_p4[1].Pt(), Jets_p4[2].Pt()))")

        df = df.Define("Max_JetsE", 
               "TMath::Max(Jets_p4[0].E(), TMath::Max(Jets_p4[1].E(), Jets_p4[2].E()))")
        df = df.Define("Min_JetsE", 
               "TMath::Min(Jets_p4[0].E(), TMath::Min(Jets_p4[1].E(), Jets_p4[2].E()))")


        # ------------------------------------------------------------------------
        # ΔR between jets
        df = df.Define("Jets_delR12", "Jets_p4[0].DeltaR(Jets_p4[1])")
        df = df.Define("Jets_delR13", "Jets_p4[0].DeltaR(Jets_p4[2])")
        df = df.Define("Jets_delR23", "Jets_p4[1].DeltaR(Jets_p4[2])")

        # Δφ between jets
        df = df.Define("Jets_delphi12", "Jets_p4[0].DeltaPhi(Jets_p4[1])")
        df = df.Define("Jets_delphi13", "Jets_p4[0].DeltaPhi(Jets_p4[2])")
        df = df.Define("Jets_delphi23", "Jets_p4[1].DeltaPhi(Jets_p4[2])")

        # Δη between jets
        df = df.Define("Jets_deleta12", "TMath::Abs(Jet1_Eta - Jet2_Eta)")
        df = df.Define("Jets_deleta13", "TMath::Abs(Jet1_Eta - Jet3_Eta)")
        df = df.Define("Jets_deleta23", "TMath::Abs(Jet2_Eta - Jet3_Eta)")

        # Δy (rapidity) between jets
        df = df.Define("Jets_delrapi12", "TMath::Abs(Jet1_Rapidity - Jet2_Rapidity)")
        df = df.Define("Jets_delrapi13", "TMath::Abs(Jet1_Rapidity - Jet3_Rapidity)")
        df = df.Define("Jets_delrapi23", "TMath::Abs(Jet2_Rapidity - Jet3_Rapidity)")

        # Δθ and angular variables between jets
        df = df.Define("Jets_deltheta12", "Jet1_Theta - Jet2_Theta")
        df = df.Define("Jets_deltheta13", "Jet1_Theta - Jet3_Theta")
        df = df.Define("Jets_deltheta23", "Jet2_Theta - Jet3_Theta")
        df = df.Define("Jets_angle12", "Jets_p4[0].Angle(Jets_p4[1].Vect())")
        df = df.Define("Jets_angle13", "Jets_p4[0].Angle(Jets_p4[2].Vect())")
        df = df.Define("Jets_angle23", "Jets_p4[1].Angle(Jets_p4[2].Vect())")
        df = df.Define("Jets_cosangle12", "TMath::Cos(Jets_angle12)")
        df = df.Define("Jets_cosangle13", "TMath::Cos(Jets_angle13)")
        df = df.Define("Jets_cosangle23", "TMath::Cos(Jets_angle23)")
        df = df.Define("Max_CosJets", "TMath::Max(Jets_cosangle12, TMath::Max(Jets_cosangle13, Jets_cosangle23))")
        df = df.Define("Min_CosJets", "TMath::Min(Jets_cosangle12, TMath::Min(Jets_cosangle13, Jets_cosangle23))")

        # HT (scalar sum of jet pT)
        df = df.Define("HT", "Jet1_Pt + Jet2_Pt + Jet3_Pt")

        # Jet–jet pair observables
        df = df.Define("JJ_M12", "(Jets_p4[0] + Jets_p4[1]).M()")
        df = df.Define("JJ_M13", "(Jets_p4[0] + Jets_p4[2]).M()")
        df = df.Define("JJ_M23", "(Jets_p4[1] + Jets_p4[2]).M()")
        df = df.Define("JJ_Mt12", "(Jets_p4[0] + Jets_p4[1]).Mt()")
        df = df.Define("JJ_Mt13", "(Jets_p4[0] + Jets_p4[2]).Mt()")
        df = df.Define("JJ_Mt23", "(Jets_p4[1] + Jets_p4[2]).Mt()")
        df = df.Define("JJ_E12", "(Jets_p4[0] + Jets_p4[1]).E()")
        df = df.Define("JJ_E13", "(Jets_p4[0] + Jets_p4[2]).E()")
        df = df.Define("JJ_E23", "(Jets_p4[1] + Jets_p4[2]).E()")

        # Jet pair pT and rapidity
        df = df.Define("jjj_PT", "Jet1_Pt + Jet2_Pt + Jet3_Pt")
        df = df.Define("jjj_y", "(Jets_p4[0] + Jets_p4[1] + Jets_p4[2]).Rapidity()")
        df = df.Define("jjj_Phi", "(Jets_p4[0] + Jets_p4[1] + Jets_p4[2]).Phi()")

        
        df = df.Define("Planarity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculatePlanarity(Jet2_P3,Jet3_P3,Jet1_P3)")
        df = df.Define("APlanarity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAplanarity(Jet2_P3,Jet3_P3,Jet1_P3)")
        df = df.Define("Sphericity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateSphericity(Jet2_P3,Jet3_P3,Jet1_P3)")
        df = df.Define("ASphericity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAsphericity(Jet2_P3,Jet3_P3,Jet1_P3)")
                
        # Individual jet scores
        df = df.Define("scoreG1", "recojet_isG[0]")
        df = df.Define("scoreG2", "recojet_isG[1]")
        df = df.Define("scoreG3", "recojet_isG[2]")

        df = df.Define("scoreU1", "recojet_isU[0]")
        df = df.Define("scoreU2", "recojet_isU[1]")
        df = df.Define("scoreU3", "recojet_isU[2]")

        df = df.Define("scoreS1", "recojet_isS[0]")
        df = df.Define("scoreS2", "recojet_isS[1]")
        df = df.Define("scoreS3", "recojet_isS[2]")

        df = df.Define("scoreC1", "recojet_isC[0]")
        df = df.Define("scoreC2", "recojet_isC[1]")
        df = df.Define("scoreC3", "recojet_isC[2]")

        df = df.Define("scoreB1", "recojet_isB[0]")
        df = df.Define("scoreB2", "recojet_isB[1]")
        df = df.Define("scoreB3", "recojet_isB[2]")

        df = df.Define("scoreT1", "recojet_isTAU[0]")
        df = df.Define("scoreT2", "recojet_isTAU[1]")
        df = df.Define("scoreT3", "recojet_isTAU[2]")

        df = df.Define("scoreD1", "recojet_isD[0]")
        df = df.Define("scoreD2", "recojet_isD[1]")
        df = df.Define("scoreD3", "recojet_isD[2]")

        # Sum of scores
        df = df.Define("scoreSumG", "recojet_isG[0] + recojet_isG[1] + recojet_isG[2]")
        df = df.Define("scoreSumU", "recojet_isU[0] + recojet_isU[1] + recojet_isU[2]")
        df = df.Define("scoreSumS", "recojet_isS[0] + recojet_isS[1] + recojet_isS[2]")
        df = df.Define("scoreSumC", "recojet_isC[0] + recojet_isC[1] + recojet_isC[2]")
        df = df.Define("scoreSumB", "recojet_isB[0] + recojet_isB[1] + recojet_isB[2]")
        df = df.Define("scoreSumT", "recojet_isTAU[0] + recojet_isTAU[1] + recojet_isTAU[2]")
        df = df.Define("scoreSumD", "recojet_isD[0] + recojet_isD[1] + recojet_isD[2]")

        # Multiply of scores
        df = df.Define("scoreMultiplyG", "recojet_isG[0] * recojet_isG[1] * recojet_isG[2]")
        df = df.Define("scoreMultiplyU", "recojet_isU[0] * recojet_isU[1] * recojet_isU[2]")
        df = df.Define("scoreMultiplyS", "recojet_isS[0] * recojet_isS[1] * recojet_isS[2]")
        df = df.Define("scoreMultiplyC", "recojet_isC[0] * recojet_isC[1] * recojet_isC[2]")
        df = df.Define("scoreMultiplyB", "recojet_isB[0] * recojet_isB[1] * recojet_isB[2]")
        df = df.Define("scoreMultiplyT", "recojet_isTAU[0] * recojet_isTAU[1] * recojet_isTAU[2]")
        df = df.Define("scoreMultiplyD", "recojet_isD[0] * recojet_isD[1] * recojet_isD[2]")

        
# __________________________________________________________________________________________

        return df
# __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = [
   
    "Iso_Photon_P",
    "Iso_Photon_Pt",
    "Iso_Photon_Eta",
    "Iso_Photon_Phi",
    "Iso_Photon_Rapidity",
    "Iso_Photon_Theta",
    "Iso_Photon_M",
    "Iso_Photon_Mt",
    "Iso_Photon_E",
    "Iso_Photon_Et",
    "Iso_Photon_CosTheta",
    "Iso_Photon_CosPhi",
    "Iso_Photons_No",

    "Iso_Electrons_No",
    "Iso_Muons_No",

    "Missing_P",
    "Missing_Pt",
    "Missing_Eta",
    "Missing_Phi",
    "Missing_Rapidity",
    "Missing_Theta",
    "Missing_M",
    "Missing_Mt",
    "Missing_E",
    "Missing_Et",
    "Missing_CosTheta",
    "Missing_CosPhi",
    "d34",
    "d45",

    #"Jets_charge",
    "MC_PrimaryVertex",
    "MC_PrimaryVertex_TLorentz",
    "displacementdz0",
    "displacementdxy0",
    "displacementdz1",
    "displacementdxy1",
     "m_jj",
    "Jet1_P",
    "Jet1_Pt",
    "Jet1_Eta",
    "Jet1_Rapidity",
    "Jet1_Phi",
    "Jet1_M",
    "Jet1_Mt",
    "Jet1_E",
    "Jet1_Et",
    "Jet1_Theta",
    "Jet1_CosTheta",
    "Jet1_CosPhi",

    "Jet2_P",
    "Jet2_Pt",
    "Jet2_Eta",
    "Jet2_Rapidity",
    "Jet2_Phi",
    "Jet2_M",
    "Jet2_Mt",
    "Jet2_E",
    "Jet2_Et",
    "Jet2_Theta",
    "Jet2_CosTheta",
    "Jet2_CosPhi",

    "Jet3_P",
    "Jet3_Pt",
    "Jet3_Eta",
    "Jet3_Rapidity",
    "Jet3_Phi",
    "Jet3_M",
    "Jet3_Mt",
    "Jet3_E",
    "Jet3_Et",
    "Jet3_Theta",
    "Jet3_CosTheta",
    "Jet3_CosPhi",
    "Max_JetsPT",
    "Min_JetsPT",
    "Max_JetsE",
    "Min_JetsE",
    "Jets_delR12",
    "Jets_delR13",
    "Jets_delR23",
    "Jets_delphi12",
    "Jets_delphi13",
    "Jets_delphi23",
    "Jets_deleta12",
    "Jets_deleta13",
    "Jets_deleta23",
    "Jets_delrapi12",
    "Jets_delrapi13",
    "Jets_delrapi23",
    "Jets_deltheta12",
    "Jets_deltheta13",
    "Jets_deltheta23",
    "Jets_angle12",
    "Jets_angle13",
    "Jets_angle23",
    "Jets_cosangle12",
    "Jets_cosangle13",
    "Jets_cosangle23",
    "Max_CosJets",
    "Min_CosJets",
    "HT",
    "JJ_M12",
    "JJ_M13",
    "JJ_M23",
    "JJ_Mt12",
    "JJ_Mt13",
    "JJ_Mt23",
    "JJ_E12",
    "JJ_E13",
    "JJ_E23",
    "jjj_PT",
    "jjj_y",
    "jjj_Phi",


    "Planarity",
    "APlanarity",
    "Sphericity",
    "ASphericity",
    "scoreG1",
    "scoreG2",
    "scoreG3",
    "scoreU1",
    "scoreU2",
    "scoreU3",
    "scoreS1",
    "scoreS2",
    "scoreS3",
    "scoreC1",
    "scoreC2",
    "scoreC3",
    "scoreB1",
    "scoreB2",
    "scoreB3",
    "scoreT1",
    "scoreT2",
    "scoreT3",
    "scoreD1",
    "scoreD2",
    "scoreD3",
    "scoreSumG",
    "scoreSumU",
    "scoreSumS",
    "scoreSumC",
    "scoreSumB",
    "scoreSumT",
    "scoreSumD",
    "scoreMultiplyG",
    "scoreMultiplyU",
    "scoreMultiplyS",
    "scoreMultiplyC",
    "scoreMultiplyB",
    "scoreMultiplyT",
    "scoreMultiplyD"
]


        return branchList
