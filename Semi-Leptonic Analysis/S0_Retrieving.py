import os, copy

# list of processes
processList = {
    'wzp6_ee_Hlnuqq_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_Hlnuqq_ecm125'},
    'wzp6_ee_Hqqlnu_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_Hqqlnu_ecm125'},
    'wzp6_ee_qq_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_qq_ecm125'},
    'wzp6_ee_eenunu_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_eenunu_ecm125'},
    'wzp6_ee_enueqq_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_enueqq_ecm125'},
    'wzp6_ee_Hgg_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_Hgg_ecm125'},
    'wzp6_ee_Hllnunu_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_Hllnunu_ecm125'},
    'wzp6_ee_l1l2nunu_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_l1l2nunu_ecm125'},
    'wzp6_ee_mumununu_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_mumununu_ecm125'},
    'wzp6_ee_munumuqq_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_munumuqq_ecm125'},
    'wzp6_ee_taunutauqq_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_taunutauqq_ecm125'},
    'wzp6_ee_tautaununu_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_tautaununu_ecm125'},

}

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
#prodTag     = "FCCee/winter2023/IDEA/"

#Optional: output directory, default is local running directory
outputDir   = "./outputs/treemaker/"
# outputDirEos = "/eos/users/r/rjafaris" #helps the output to be visible in CERNbox (does not work!)


# Define the input dir (optional)
#inputDir    = "./localSamples/"
inputDir    = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"

# additional/costom C++ functions, defined in header files (optional)
includePaths = ["functions.h"]

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

# __________________________________________________________
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):

 # __________________________________________________________
        # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2

        # define some aliases to be used later on
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Electron0", "Electron#0.index")
        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Photon0", "Photon#0.index")

        # __________________________________________________________				
        # get all the leptons from the collection
        df = df.Define("electrons_all", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)",)
        df = df.Define("muons_all", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)",)
        df = df.Define("photons_all", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)",)
 # ______________________________________________________________________________
        # select leptons with momentum > 0 GeV
        #df = df.Define("electrons", "FCCAnalyses::ReconstructedParticle::sel_p(20)(electrons_all)",)
        df = df.Define("electrons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(electrons_all)",)
        df = df.Define("muons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(muons_all)",)
        df = df.Define("photons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(photons_all)",)

 # ______________________________________________________________________________

 # Lepton Isolation:
 # ______________________________________________________________________________
        # compute the electron isolation and store leptons with an isolation cut of 0df = df.25 in a separate column electrons_sel_iso
        df = df.Define("electrons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(electrons, ReconstructedParticles)",)
        df = df.Define( "electrons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.25)(electrons, electrons_iso)",)
        #df = df.Define("isoelectrons_no", "electrons_sel_iso.size()")
        df = df.Define("isoelectrons_no",  "FCCAnalyses::ReconstructedParticle::get_n(electrons_sel_iso)")
#----- Two above lines give exactly a same output, but with different bin intervals; We chose the 2nd.  

        df = df.Define("muons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(muons, ReconstructedParticles)",)
        df = df.Define("muons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.25)(muons, muons_iso)",)
        #df = df.Define("isomuons_no", "muons_sel_iso.size()")
        df = df.Define("isomuons_no",  "FCCAnalyses::ReconstructedParticle::get_n(muons_sel_iso)")

 # Photon Isolation: (to check if there is any isolated photon or not, see: isophotons_no)
 # ______________________________________________________________________________
        df = df.Define("photons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(photons, ReconstructedParticles)",)
        df = df.Define( "photons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.25)(photons, photons_iso)",)
        #df = df.Define("isophotons_no", "photons_sel_iso.size()")
        df = df.Define("isophotons_no",  "FCCAnalyses::ReconstructedParticle::get_n(photons_sel_iso)")

## ==================================== leptons =================================================

 # elec:_________________________________________________________________________
        df = df.Define("electrons_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons)")
        df = df.Define("electrons_pt", "FCCAnalyses::ReconstructedParticle::get_pt(electrons)")
        df = df.Define("electrons_e", "FCCAnalyses::ReconstructedParticle::get_e(electrons)")
        df = df.Define("electrons_eta",   "FCCAnalyses::ReconstructedParticle::get_eta(electrons)",)
        df = df.Define("electrons_theta",   "FCCAnalyses::ReconstructedParticle::get_theta(electrons)",)
        df = df.Define("electrons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(electrons)",)
        df = df.Define("electrons_q",   "FCCAnalyses::ReconstructedParticle::get_charge(electrons)",)
        df = df.Define("electrons_no",   "FCCAnalyses::ReconstructedParticle::get_n(electrons)",)

 # muon:_________________________________________________________________________
        df = df.Define("muons_p", "FCCAnalyses::ReconstructedParticle::get_p(muons)")
        df = df.Define("muons_pt", "FCCAnalyses::ReconstructedParticle::get_pt(muons)")
        df = df.Define("muons_e", "FCCAnalyses::ReconstructedParticle::get_e(muons)")
        df = df.Define("muons_eta",   "FCCAnalyses::ReconstructedParticle::get_eta(muons)",)
        df = df.Define("muons_theta",   "FCCAnalyses::ReconstructedParticle::get_theta(muons)",)
        df = df.Define("muons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(muons)",)
        df = df.Define("muons_q",   "FCCAnalyses::ReconstructedParticle::get_charge(muons)",)
        df = df.Define("muons_no",  "FCCAnalyses::ReconstructedParticle::get_n(muons)")


## ==================================== isolated leptons =================================================

 # iso elec:_________________________________________________________________________
        df = df.Define("isoelectrons_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons_sel_iso)")
        df = df.Define("isoelectrons_pt", "FCCAnalyses::ReconstructedParticle::get_pt(electrons_sel_iso)")
        df = df.Define("isoelectrons_e", "FCCAnalyses::ReconstructedParticle::get_e(electrons_sel_iso)")
        df = df.Define("isoelectrons_eta",   "FCCAnalyses::ReconstructedParticle::get_eta(electrons_sel_iso)")
        df = df.Define("isoelectrons_theta",   "FCCAnalyses::ReconstructedParticle::get_theta(electrons_sel_iso)")
        df = df.Define("isoelectrons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(electrons_sel_iso)")
        df = df.Define("isoelectrons_q",   "FCCAnalyses::ReconstructedParticle::get_charge(electrons_sel_iso)")

 # iso muon:_________________________________________________________________________
        df = df.Define("isomuons_p", "FCCAnalyses::ReconstructedParticle::get_p(muons_sel_iso)")
        df = df.Define("isomuons_pt", "FCCAnalyses::ReconstructedParticle::get_pt(muons_sel_iso)")
        df = df.Define("isomuons_e", "FCCAnalyses::ReconstructedParticle::get_e(muons_sel_iso)")
        df = df.Define("isomuons_eta",   "FCCAnalyses::ReconstructedParticle::get_eta(muons_sel_iso)")
        df = df.Define("isomuons_theta",   "FCCAnalyses::ReconstructedParticle::get_theta(muons_sel_iso)")
        df = df.Define("isomuons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(muons_sel_iso)")
        df = df.Define("isomuons_q",   "FCCAnalyses::ReconstructedParticle::get_charge(muons_sel_iso)")

        df = df.Define("isoleptons_no", "isomuons_no + isoelectrons_no")

# how to define leptons_sel_iso ?

 # ______________________________________________________________________________

        #########
        ### CUT 1: at least 1 electron with at least one isolated one
        #########
        #df = df.Filter("leptons_sel_iso.size() > 0") # to check and see the plot to check if we really have 1 iso lep; After that we require exactly one iso lep.

        df = df.Filter("(electrons_sel_iso.size() > 0 && muons_sel_iso.size() == 0) || (electrons_sel_iso.size() == 0 && muons_sel_iso.size() > 0)")

        #df = df.Filter("electrons_sel_iso.size() > 0 || muons_sel_iso.size() > 0")


        #df = df.Filter("photons_sel_iso.size() == 0")

        #########
        ### CUT 1: at least 1 electron with at least one isolated one
        #########
        #df = df.Filter("electrons_no >= 1 && electrons_sel_iso.size() > 0")


        #########
        ### CUT 2 :at least 2 opposite-sign (OS) leptons
        #########
    #    df = df.Filter("electrons_no >= 2 && abs(Sum(electrons_q)) < electrons_q.size()")
        # now we build the Z resonance based on the available leptons.
        # the function resonanceBuilder_mass_recoil returns the best lepton pair compatible with the Z mass (91.2 GeV) and recoil at 125 GeV
        # the argument 0.4 gives a weight to the Z mass and the recoil mass in the chi2 minimization
        # technically, it returns a ReconstructedParticleData object with index 0 the di-lepton system, index and 2 the leptons of the pair

        ## here cluster jets in the events but first remove electrons from the list of
        ## reconstructed particles

 # __________________________Required for JET CLUSTERING________________________________

        ## create a new collection of reconstructed particles removing electrons with p>20
        df = df.Define("ReconstructedParticlesNoElectrons",
                        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles,electrons)",)

        df = df.Define("ReconstructedParticlesNoleptons",
                        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoElectrons,muons)",)

 # _______________________________JET CLUSTERING_________________________________

        ## perform N=2 jet clustering
        global jetClusteringHelper
        global jetFlavourHelper


        ## define jet and run clustering parameters
        ## name of collections in EDM root files
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

        collections_noleptons = copy.deepcopy(collections)
        collections_noleptons["PFParticles"] = "ReconstructedParticlesNoleptons"

        jetClusteringHelper = ExclusiveJetClusteringHelper(collections_noleptons["PFParticles"], 2)
        df = jetClusteringHelper.define(df)
#_________________________________________________________________

# flavor tagging removed from here (check the tutorial if you need it).

# __________________________________________________________________________________________

        df = df.Define("missingEnergy", "FCCAnalyses::ZHfunctions::missingEnergy(125., ReconstructedParticles)",)
        # .Define("cosTheta_miss", "FCCAnalyses::get_cosTheta_miss(missingEnergy)")
        df = df.Define("cosTheta_miss", "FCCAnalyses::ZHfunctions::get_cosTheta_miss(MissingET)",)
        df = df.Define("missing_p", "FCCAnalyses::ReconstructedParticle::get_p(MissingET)",)

        #########
        ### CUT 3: Njets = 2
        #########

        df = df.Filter("event_njet > 1")

        ## define jet clustering parameters N = 2
        #jetClusteringHelper_N2 = ExclusiveJetClusteringHelper("ReconstructedParticles", 2, "N2") # ASK WHY MICHELE USE THIS?

        jetClusteringHelper_N2 = ExclusiveJetClusteringHelper("ReconstructedParticlesNoleptons", 2, "N2") 
        jetClusteringHelper_N3 = ExclusiveJetClusteringHelper("ReconstructedParticlesNoleptons", 3, "N3")

        # jetClusteringHelper_N2 = ExclusiveJetClusteringHelper("ReconstructedParticles_noiso", 2, "N2")
        df = jetClusteringHelper_N2.define(df)
        df = jetClusteringHelper_N3.define(df)

        df = df.Define("d23", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 2))")  # dmerge from 3 to 2
        df = df.Define("d34", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 3))")  # dmerge from 4 to 3


        df = df.Define("jets_p4",   "JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets),)
        df = df.Define("IM",  "JetConstituentsUtils::InvariantMass(jets_p4[0], jets_p4[1])",)

        #df = df.Define("jet1_p4", "jets_p4[0]")
        #df = df.Define("jet2_p4", "jets_p4[1]")
# __________________________________________________________________________________________

       # Define kinematic variables for the jets

## we don't need to define jet_n, because it will be produced automatiacally by: branchList += jetClusteringHelper.outputBranches()
        df = df.Define("jet1_pt", "jets_p4[0].Pt()")
        df = df.Define("jet1_eta", "jets_p4[0].Eta()")
        df = df.Define("jet1_phi", "jets_p4[0].Phi()")
        df = df.Define("jet1_mass", "jets_p4[0].M()")
        df = df.Define("jet1_e", "jets_p4[0].E()")
        df = df.Define("jet1_p", "jets_p4[0].P()")
        df = df.Define("jet1_theta", "jets_p4[0].Theta()")

        df = df.Define("jet2_pt", "jets_p4[1].Pt()")
        df = df.Define("jet2_eta", "jets_p4[1].Eta()")
        df = df.Define("jet2_phi", "jets_p4[1].Phi()")
        df = df.Define("jet2_mass", "jets_p4[1].M()")
        df = df.Define("jet2_e", "jets_p4[1].E()")
        df = df.Define("jet2_p", "jets_p4[1].P()")
        df = df.Define("jet2_theta", "jets_p4[1].Theta()")

# __________________________________________________________________________________________

        return df

    # __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = [

            "isophotons_no",
            "isoleptons_no",

            "electrons_no",
            "isoelectrons_no",
            "isoelectrons_p",
            "isoelectrons_pt",
            "isoelectrons_e",
            "isoelectrons_eta",
            "isoelectrons_phi",
            "isoelectrons_theta",

            "muons_no",
            "isomuons_no",
            "isomuons_p",
            "isomuons_pt",
            "isomuons_e",
            "isomuons_eta",
            "isomuons_phi",
            "isomuons_theta",

            "jet1_e",
            "jet1_p",
            "jet1_pt",
            "jet1_theta",
            "jet1_eta",
            "jet1_phi",
            "jet1_mass",

            "jet2_e",
            "jet2_p",
            "jet2_pt",
            "jet2_theta",
            "jet2_eta",
            "jet2_phi",
            "jet2_mass",    

            "cosTheta_miss",
            "missing_p",
            "missingEnergy",
            "IM",
            "d23",            
            "d34",  
                                                                     
        ]

        ##  outputs jet properties
        branchList += jetClusteringHelper.outputBranches()

        ## outputs jet scores and constituent breakdown
        #branchList += jetFlavourHelper.outputBranches()

        return branchList

