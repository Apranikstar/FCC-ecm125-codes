#Mandatory: List of processes
processList = {
    'wzp6_ee_taunutauqq_ecm125':{'fraction':1.0, 'chunks':1, 'output':'wzp6_ee_taunutauqq_ecm125'}
}

#Mandatory: Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics
prodTag     = "FCCee/winter2023/IDEA"

#Optional: output directory, default is local running directory
outputDir   = "outputs/wzp6_ee_taunutauqq_ecm125"

#Optional
nCPUS       = 8
runBatch    = False
#batchQueue = "longlunch"
#compGroup = "group_u_FCC.local_gen"

# test file
testFile ="/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/wzp6_ee_taunutauqq_ecm125/"


class RDFanalysis():

    #__________________________________________________________
    #Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):
        df2 = (df
               #############################################
               ##          Aliases for # in python        ##
               #############################################

				#PHOTONS
				.Alias("Photon0", "Photon#0.index")
				.Define("photons",  "ReconstructedParticle::get(Photon0, ReconstructedParticles)")
				.Define("n_photons",  "ReconstructedParticle::get_n(photons)") #count how many photons are in the event in total
				.Define("photons_pT",     "ReconstructedParticle::get_pt(photons)") #transverse momentum pT
				.Define("photons_eta",     "ReconstructedParticle::get_eta(photons)") #pseudorapidity eta
				.Define("photons_phi",     "ReconstructedParticle::get_phi(photons)") #polar angle in the transverse plane phi
				.Define("photons_mass",     "ReconstructedParticle::get_mass(photons)") #polar angle in the transverse plane phi
				.Define("photons_energy",     "ReconstructedParticle::get_e(photons)") #polar angle in the transverse plane phi
				.Define("photons_charge",     "ReconstructedParticle::get_charge(photons)") #polar angle in the transverse plane phi
				.Define("photons_p",     "ReconstructedParticle::get_p(photons)") #polar angle in the transverse plane phi
				.Define("photons_px",     "ReconstructedParticle::get_px(photons)") #polar angle in the transverse plane phi
				.Define("photons_py",     "ReconstructedParticle::get_py(photons)") #polar angle in the transverse plane phi
				.Define("photons_pz",     "ReconstructedParticle::get_pz(photons)") #polar angle in the transverse plane phi
				.Define("photons_rapidity",     "ReconstructedParticle::get_y(photons)") #polar angle in the transverse plane phi
				.Define("photons_theta",     "ReconstructedParticle::get_theta(photons)") #polar angle in the transverse plane phi
				#.Define("photons_TLV",     "ReconstructedParticle::get_tlv(photons)") #polar angle in the transverse plane phi



				#ELECTRONS Kinematics

				.Alias("Electron0", "Electron#0.index")
				.Define("electrons",  "ReconstructedParticle::get(Electron0, ReconstructedParticles)")
				.Define("n_electrons",  "ReconstructedParticle::get_n(electrons)") #count how many electrons are in the event in total
				.Define("electrons_pT",     "ReconstructedParticle::get_pt(electrons)") #transverse momentum pT
				.Define("electrons_eta",     "ReconstructedParticle::get_eta(electrons)") #pseudorapidity eta
				.Define("electrons_phi",     "ReconstructedParticle::get_phi(electrons)") #polar angle in the transverse plane phi
				.Define("electrons_mass",     "ReconstructedParticle::get_mass(electrons)") #mass
				.Define("electrons_energy",     "ReconstructedParticle::get_e(electrons)")
				.Define("electrons_charge",     "ReconstructedParticle::get_charge(electrons)") #polar angle in the transverse plane phi
				.Define("electrons_p",     "ReconstructedParticle::get_p(electrons)") #polar angle in the transverse plane phi
				.Define("electrons_px",     "ReconstructedParticle::get_px(electrons)") #polar angle in the transverse plane phi
				.Define("electrons_py",     "ReconstructedParticle::get_py(electrons)") #polar angle in the transverse plane phi
				.Define("electrons_pz",     "ReconstructedParticle::get_pz(electrons)") 
				.Define("electrons_rapidity",     "ReconstructedParticle::get_y(electrons)") #
				.Define("electrons_theta",     "ReconstructedParticle::get_theta(electrons)") 
				#.Define("electrons_TLV",     "ReconstructedParticle::get_tlv(electrons)") 
				# Kinematics of Muons
				.Alias("Muon0", "Muon#0.index")
				.Define("muons",  "ReconstructedParticle::get(Muon0, ReconstructedParticles)")
				.Define("n_muons",  "ReconstructedParticle::get_n(muons)") #count how many muons are in the event in total
				.Define("muons_pT",     "ReconstructedParticle::get_pt(muons)") #transverse momentum pT
				.Define("muons_eta",     "ReconstructedParticle::get_eta(muons)") #pseudorapidity eta
				.Define("muons_phi",     "ReconstructedParticle::get_phi(muons)") #polar angle in the transverse plane phi
				.Define("muons_mass",     "ReconstructedParticle::get_mass(muons)")
				.Define("muons_energy",     "ReconstructedParticle::get_e(muons)")
				.Define("muons_charge",     "ReconstructedParticle::get_charge(muons)") #polar angle in the transverse plane phi
				.Define("muons_p",     "ReconstructedParticle::get_p(muons)") #polar angle in the transverse plane phi
				.Define("muons_px",     "ReconstructedParticle::get_px(muons)") #polar angle in the transverse plane phi
				.Define("muons_py",     "ReconstructedParticle::get_py(muons)") #polar angle in the transverse plane phi
				.Define("muons_pz",     "ReconstructedParticle::get_pz(muons)") 
				.Define("muons_rapidity",     "ReconstructedParticle::get_y(muons)") #
				.Define("muons_theta",     "ReconstructedParticle::get_theta(muons)") 
				#.Define("muons_TLV",     "ReconstructedParticle::get_tlv(muons)") 

				#OBJECT SELECTION: Consider only those objects that have pT > certain threshold
				#.Define("selected_jets", "ReconstructedParticle::sel_pt(50.)(Jet)") #select only jets with a pT > 50 GeV
				#.Define("selected_electrons", "ReconstructedParticle::sel_pt(20.)(electrons)") #select only electrons with a pT > 20 GeV
				#.Define("selected_muons", "ReconstructedParticle::sel_pt(20.)(muons)")

				#SIMPLE VARIABLES: Access the basic kinematic variables of the selected jets, works analogously for electrons, muons
				#.Define("selected_jets", "ReconstructedParticle::sel_pt(20.)(Jet)") #select only jets with a pT > 50 GeV
				.Define("n_jets", "ReconstructedParticle::get_n(Jet)") #count how many jets are in the event in total
				.Define("jet_pT",     "ReconstructedParticle::get_pt(Jet)") #transverse momentum pT
				.Define("jet_eta",     "ReconstructedParticle::get_eta(Jet)") #pseudorapidity eta
				.Define("jet_phi",     "ReconstructedParticle::get_phi(Jet)") #polar angle in the transverse plane phi
				.Define("jet_rapidity",     "ReconstructedParticle::get_y(Jet)") #polar angle in the transverse plane phi
				.Define("jet_theta",     "ReconstructedParticle::get_theta(Jet)") #polar angle in the transverse plane phi
				.Define("jet_energy",     "ReconstructedParticle::get_e(Jet)")
				#EVENTWIDE VARIABLES: Access quantities that exist only once per event, such as the missing transverse energy
				.Define("MET_energy", "ReconstructedParticle::get_e(MissingET)") #energy value of MET
				.Define("MET_pT", "ReconstructedParticle::get_pt(MissingET)") # pT value of MET (absolute)
				.Define("MET_px", "ReconstructedParticle::get_px(MissingET)") #x-component of MET
				.Define("MET_py", "ReconstructedParticle::get_py(MissingET)") #y-component of MET
				.Define("MET_pz", "ReconstructedParticle::get_pz(MissingET)") #y-component of MET
				.Define("MET_phi", "ReconstructedParticle::get_phi(MissingET)") #phi of MET
				.Define("MET_eta", "ReconstructedParticle::get_eta(MissingET)") #angle of MET
				.Define("MET_mass", "ReconstructedParticle::get_mass(MissingET)") #angle of MET

              )
        return df2

    #__________________________________________________________
    #Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = [
	   "n_photons",
	   "photons_pT",
	   "photons_eta",
	   "photons_phi",
	   "photons_mass",
	   "photons_energy",
	   "photons_charge",
	   "photons_p",
	   "photons_px",
	   "photons_py",
	   "photons_pz",
	   "photons_rapidity",
	   "photons_theta",
	   #"photons_TLV",
	   "n_electrons",
	   "electrons_pT",
	   "electrons_eta",
	   "electrons_phi",
	   "electrons_mass", 
	   "electrons_energy", 
	   "electrons_charge",
	   "electrons_p",
	   "electrons_px",
	   "electrons_py",
	   "electrons_pz",
	   "electrons_rapidity",
	   "electrons_theta",
	   #"electrons_TLV",
	   "n_muons",
	   "muons_pT",
	   "muons_eta",
	   "muons_phi",
	   "muons_mass",
	   "muons_energy",
	   "muons_charge",
	   "muons_p",
	   "muons_px",
	   "muons_py",
	   "muons_pz",
	   "muons_rapidity",
	   "muons_theta",
	   #"muons_TLV",
	   "n_jets",
	   "jet_pT",
	   "jet_eta",
	   "jet_phi",
	   "jet_rapidity",
	   "jet_theta",
	   "jet_energy",  
	   "MET_energy",
	   "MET_pT",
	   "MET_px",
	   "MET_py",
	   "MET_pz",
	   "MET_phi",
	   "MET_eta",
	   "MET_mass",
	    ]
        return branchList
