import ROOT
import os
from ROOT import TCut
from ROOT import TMath
import gc

######################################
TMVA = ROOT.TMVA
TFile = ROOT.TFile
TMVA.Tools.Instance()
useBDT = True  # Boosted Decision Tree

######################################

outputFile = TFile.Open("SL-Hlnuqq_ElectronAnalysis_.root", "RECREATE")
factory = TMVA.Factory(
    "TMVA_Classification", outputFile, V=False, ROC=True, Silent=False, Color=True, AnalysisType="Classification"
)
loader = TMVA.DataLoader("Electron_DataSet")

#################################################
#SGNL
#Electron_wzp6_ee_eenunu_ecm125.root
#Electron_wzp6_ee_enueqq_ecm125.root
#Electron_wzp6_ee_Hgg_ecm125.root
#Electron_wzp6_ee_Hllnunu_ecm125.root
#Electron_wzp6_ee_Hlnuqq_ecm125.root
#Electron_wzp6_ee_Hqqlnu_ecm125.root
#Electron_wzp6_ee_l1l2nunu_ecm125.root
#Electron_wzp6_ee_mumununu_ecm125.root
#Electron_wzp6_ee_munumuqq_ecm125.root
#Electron_wzp6_ee_qq_ecm125.root
#Electron_wzp6_ee_taunutauqq_ecm125.root
#Electron_wzp6_ee_tautaununu_ecm125.root
### Signal File and Tree

HlnuqqFile = TFile.Open("Electron_DataBase/Electron_wzp6_ee_Hlnuqq_ecm125.root")
HlnuqqTree = HlnuqqFile.Get("events") ### Check

### Background Files and Trees
eenunuFile = TFile.Open("Electron_DataBase/Electron_wzp6_ee_eenunu_ecm125.root")
eenunuTree = eenunuFile.Get("events")

enueqqFile = TFile.Open("Electron_DataBase/Electron_wzp6_ee_enueqq_ecm125.root")
eenueqqTree = enueqqFile.Get("events")

munumuqqFile = TFile.Open("Electron_DataBase/Electron_wzp6_ee_munumuqq_ecm125.root")
munumuqqTree = munumuqqFile.Get("events")

taunutauqqFile = TFile.Open("Electron_DataBase/Electron_wzp6_ee_taunutauqq_ecm125.root")
taunutauqqTree = taunutauqqFile.Get("events")

tautaununuFile = TFile.Open("Electron_DataBase/Electron_wzp6_ee_tautaununu_ecm125.root")
tautaununuTree = tautaununuFile.Get("events")

l1l2nunuFile = TFile.Open("Electron_DataBase/Electron_wzp6_ee_l1l2nunu_ecm125.root")
l1l2nunuTree = l1l2nunuFile.Get("events")

qqFile = TFile.Open("Electron_DataBase/Electron_wzp6_ee_qq_ecm125.root")
qqTree = qqFile.Get("events")

HllnunuFile = TFile.Open("Electron_DataBase/Electron_wzp6_ee_Hllnunu_ecm125.root")
HlnunuTree = HllnunuFile.Get("events")

HggFile = TFile.Open("Electron_DataBase/Electron_wzp6_ee_Hgg_ecm125.root")
HggTree = HggFile.Get("events")


############### Signal and Backgrounds, Modify as needed

loader.AddSignalTree(HlnuqqTree, 1.03e-16)
######################################
loader.AddBackgroundTree(eenunuTree, 5.6e-14)  
loader.AddBackgroundTree(eenueqqTree, 9.98e-14) 

loader.AddBackgroundTree(munumuqqTree, 1.58e-15)

loader.AddBackgroundTree(taunutauqqTree,9.33e-15)
loader.AddBackgroundTree(tautaununuTree, 1.56e-14)

loader.AddBackgroundTree(l1l2nunuTree,2.23e-14)

loader.AddBackgroundTree(qqTree, 2.95e-10)

loader.AddBackgroundTree(HlnunuTree,7.48e-17)
loader.AddBackgroundTree(HggTree,8.1e-18)

######################################


######### Variables
loader.AddVariable("Delta_rapidity_JNLO_JLO", "Delta_rapidity_JNLO_JLO", "F")
loader.AddVariable("Delta_rapidity_electron_JNLO", "Delta_rapidity_electron_JNLO", "F")
loader.AddVariable("Delta_rapidity_electron_JLO", "Delta_rapidity_electron_JLO", "F")
loader.AddVariable("JetNLO_Rapidity", "JetNLO_Rapidity", "F")
loader.AddVariable("JetLO_Rapidity", "JetLO_Rapidity", "F")
loader.AddVariable("electron_Rapidity", "electron_Rapidity", "F")
#loader.AddVariable("MET_Rapidity", "MET_Rapidity", "F"); # Causes Crashes
loader.AddVariable("Delta_eta_JLO_JNLO", "Delta_eta_JLO_JNLO", "F")
loader.AddVariable("Delta_eta_electron_NJO", "Delta_eta_electron_NJO", "F")
loader.AddVariable("Delta_eta_electron_JLO", "Delta_eta_electron_JLO", "F")
loader.AddVariable("JetNLO_eta", "JetNLO_eta", "F")
loader.AddVariable("JetLO_eta", "JetLO_eta", "F")
loader.AddVariable("electron_eta", "electron_eta", "F")
#loader.AddVariable("MET_eta", "MET_eta", "F")
loader.AddVariable("PT_electron", "PT_electron", "F")
loader.AddVariable("PT_JNLO", "PT_JNLO", "F")
loader.AddVariable("PT_JLO", "PT_JLO", "F")
loader.AddVariable("Energy_Transverse_electron", "Energy_Transverse_electron", "F")
loader.AddVariable("Energy_Transverse_JNLO", "Energy_Transverse_JNLO", "F")
loader.AddVariable("Energy_Transverse_JLO", "Energy_Transverse_JLO", "F")
loader.AddVariable("Energy_Transverse_met", "Energy_Transverse_met", "F")
loader.AddVariable("Energy_electron", "Energy_electron", "F")
loader.AddVariable("Energy_JNLO", "Energy_JNLO", "F")
loader.AddVariable("Energy_JLO", "Energy_JLO", "F")
loader.AddVariable("M_JLO", "M_JLO", "F")
loader.AddVariable("M_JNLO", "M_JNLO", "F")
loader.AddVariable("M_electron", "M_electron", "F")
loader.AddVariable("M_met", "M_met", "F")
loader.AddVariable("MT_electron", "MT_electron", "F")
loader.AddVariable("MT_JNLO", "MT_JNLO", "F")
loader.AddVariable("MT_JLO", "MT_JLO", "F")
loader.AddVariable("MT_met", "MT_met", "F")
loader.AddVariable("Phi_electron", "Phi_electron", "F")
loader.AddVariable("Phi_JNLO", "Phi_JNLO", "F")
loader.AddVariable("Phi_JLO", "Phi_JLO", "F")
loader.AddVariable("Theta_electron", "Theta_electron", "F")
loader.AddVariable("Theta_JNLO", "Theta_JNLO", "F")
loader.AddVariable("Theta_JLO", "Theta_JLO", "F")
loader.AddVariable("cosTheta_electron", "cosTheta_electron", "F")
loader.AddVariable("cosTheta_JNLO", "cosTheta_JNLO", "F")
loader.AddVariable("cosTheta_JLO", "cosTheta_JLO", "F")
loader.AddVariable("CosAngle_JNLO_JLO", "CosAngle_JNLO_JLO", "F")
loader.AddVariable("CosAngle_Electron_JNLO", "CosAngle_Electron_JNLO", "F")
loader.AddVariable("CosAngle_Electron_JLO", "CosAngle_Electron_JLO", "F")
loader.AddVariable("Linear_Aplanarity", "Linear_Aplanarity", "F")
loader.AddVariable("Aplanarity", "Aplanarity", "F")
loader.AddVariable("Sphericity", "Sphericity", "F")
loader.AddVariable("DeltaPhi_JNLO_JLO", "DeltaPhi_JNLO_JLO", "F")
loader.AddVariable("DeltaPhi_electron_JNLO", "DeltaPhi_electron_JNLO", "F")
loader.AddVariable("DeltaPhi_electron_JLO", "DeltaPhi_electron_JLO", "F")
loader.AddVariable("DeltaR_JNLO_JLO", "DeltaR_JNLO_JLO", "F")
loader.AddVariable("DeltaR_electron_JLO", "DeltaR_electron_JLO", "F")
loader.AddVariable("DeltaR_electron_JNLO", "DeltaR_electron_JNLO", "F")
loader.AddVariable("Energy_MET", "Energy_MET", "F")

######################################
# Universal parameters
cut1 = TCut("Energy_JLO < 52.45")
cut2 = TCut("Energy_JNLO < 52.45")
cut3 = TCut("Energy_electron > 10.0")
cut4 = TCut("M_met < 3")
cut5 = TCut("Energy_MET > 20")
combined_cuts = cut1 + cut2 + cut3 + cut4 + cut5
mycutb = combined_cuts
#####################
loader.PrepareTrainingAndTestTree(combined_cuts, mycutb, "SplitMode=Random:NormMode=NumEvents:!V") # nTrain_Signal=50%:nTrain_Background=0:
#####################
if useBDT:
    factory.BookMethod(
        loader,
        ROOT.TMVA.Types.kBDT,
        "BDT",
        "!V:NTrees=100:MinNodeSize=0.5:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.6:UseBaggedBoost=True:BaggedSampleFraction=0.6:SeparationType=GiniIndex:nCuts=30"
    )
######################
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
outputFile.Close()
# Delete objects after use
del loader
del factory
#force garbage collection
gc.collect()

