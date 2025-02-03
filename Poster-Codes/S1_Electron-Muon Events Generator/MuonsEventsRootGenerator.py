import uproot
import numpy as np
import awkward as ak
from ROOT import TLorentzVector
import matplotlib.pyplot as plt


myFileName = ""



###### Subroutines
def GetRootFile(rootFile):
    return uproot.open(rootFile)

def PreSelectionCutEventsList(RootFile):
    n_elec = np.array(RootFile["events"]["n_muons"].array())
    n_jets = np.array(RootFile["events"]["n_jets"].array())
    MET_energy = np.array(RootFile["events"]["MET_energy"].array())
    electronPreCut = np.where(n_elec == 1)[0]
    jetsPreCut = np.where(n_jets == 2)[0]
    metPreCut = np.where(MET_energy > 2)[0]
    commonEvents = np.intersect1d(np.intersect1d(electronPreCut, jetsPreCut), metPreCut)
    efficiency = np.divide(len(commonEvents) ,len(n_elec))
    return commonEvents,efficiency

def electrons4Vec(RootFile, IndexList):
    electron4Vec = []

    # Extract all necessary arrays upfront
    electron_pT = RootFile["events"]["muons_pT"].array()
    electron_eta = RootFile["events"]["muons_eta"].array()
    electron_phi = RootFile["events"]["muons_phi"].array()
    electron_energy = RootFile["events"]["muons_energy"].array()

    # Loop through enujj_common_indices efficiently
    for idx in IndexList:
    # Use the index to access the first element directly
        pT = electron_pT[idx][0]
        eta = electron_eta[idx][0]
        phi = electron_phi[idx][0]
        energy = electron_energy[idx][0]
    
        # Create TLorentzVector and compute invariant mass
        invMassE = TLorentzVector()
        invMassE.SetPtEtaPhiE(pT, eta, phi, energy)
        electron4Vec.append(invMassE)
    return electron4Vec


def jetLO4vec(RootFile, IndexList):
    JetLO4VecList = []
    # Pre-extract all necessary arrays
    jets_pT = RootFile["events"]["jet_pT"].array()
    jets_eta = RootFile["events"]["jet_eta"].array()
    jets_phi = RootFile["events"]["jet_phi"].array()
    jets_energy = RootFile["events"]["jet_energy"].array()

    # Loop through enujj_common_indices
    for idx in IndexList:
        # Use the index to access the first element directly
        pT = jets_pT[idx][0]
        eta = jets_eta[idx][0]
        phi = jets_phi[idx][0]
        energy = jets_energy[idx][0]

        # Create TLorentzVector and compute invariant mass
        jetLO4Vec = TLorentzVector()
        jetLO4Vec.SetPtEtaPhiE(pT, eta, phi, energy)
        JetLO4VecList.append(jetLO4Vec)
    return JetLO4VecList

def jetNLO4vec(RootFile, IndexList):
    JetNLO4VecList = []
    # Pre-extract all necessary arrays
    jets_pT = RootFile["events"]["jet_pT"].array()
    jets_eta = RootFile["events"]["jet_eta"].array()
    jets_phi = RootFile["events"]["jet_phi"].array()
    jets_energy = RootFile["events"]["jet_energy"].array()

    # Loop through enujj_common_indices
    for idx in IndexList:
        # Use the index to access the first element directly
        pT = jets_pT[idx][1]
        eta = jets_eta[idx][1]
        phi = jets_phi[idx][1]
        energy = jets_energy[idx][1]

        # Create TLorentzVector and compute invariant mass
        jetNLO4Vec = TLorentzVector()
        jetNLO4Vec.SetPtEtaPhiE(pT, eta, phi, energy)
        JetNLO4VecList.append(jetNLO4Vec)
    return JetNLO4VecList


def MET4Vec(RootFile, IndexList):
    MET4VecList = []
    # Pre-extract all necessary arrays
    met_Px = RootFile["events"]["MET_px"].array()
    met_Py = RootFile["events"]["MET_py"].array()
    met_Pz = RootFile["events"]["MET_pz"].array()
    met_Energy = RootFile["events"]["MET_energy"].array()

    # Loop through enujj_common_indices
    for idx in IndexList:
        # Use the index to access the first element directly
        Px = met_Px[idx][0]
        Py = met_Py[idx][0]
        Pz = met_Pz[idx][0]
        energy = met_Energy[idx][0]

        # Create TLorentzVector and compute invariant mass
        met4Vec = TLorentzVector()
        met4Vec.SetPxPyPzE(Px, Py, Pz, energy)
        MET4VecList.append(met4Vec)
    return MET4VecList


def compute_sphericity(vec1, vec2, vec3):
    """
    Computes the sphericity for an event with exactly 3 TLorentzVector objects.
    
    Parameters:
        vec1, vec2, vec3: ROOT.TLorentzVector
            The 3 TLorentzVector objects representing particles in the event.
    
    Returns:
        float: The sphericity value (0 ≤ S ≤ 1).
    """
    # Step 1: Initialize sphericity tensor and total momentum sum squared
    sphericity_tensor = np.zeros((3, 3))
    momentum_sum_squared = 0.0

    # List of vectors
    particles = [vec1, vec2, vec3]

    for p in particles:
        px, py, pz = p.Px(), p.Py(), p.Pz()
        momentum = np.array([px, py, pz])
        momentum_sum_squared += np.dot(momentum, momentum)  # |p|^2
        sphericity_tensor += np.outer(momentum, momentum)  # p_i * p_j

    # Step 2: Normalize the sphericity tensor
    sphericity_tensor /= momentum_sum_squared

    # Step 3: Compute eigenvalues and sort them in descending order
    eigenvalues = np.linalg.eigvals(sphericity_tensor)
    eigenvalues = np.sort(eigenvalues)[::-1]  # λ1 >= λ2 >= λ3

    # Step 4: Calculate sphericity
    sphericity = 1.5 * (eigenvalues[1] + eigenvalues[2])  # λ2 + λ3

    return sphericity



def compute_aplanarity(vec1, vec2, vec3):
    """
    Computes the aplanarity for an event with exactly 3 TLorentzVector objects.
    
    Parameters:
        vec1, vec2, vec3: ROOT.TLorentzVector
            The 3 TLorentzVector objects representing particles in the event.
    
    Returns:
        float: The aplanarity value (0 ≤ A ≤ 0.5).
    """
    # Step 1: Initialize sphericity tensor and total momentum sum squared
    sphericity_tensor = np.zeros((3, 3))
    momentum_sum_squared = 0.0

    # List of vectors
    particles = [vec1, vec2, vec3]

    for p in particles:
        px, py, pz = p.Px(), p.Py(), p.Pz()
        momentum = np.array([px, py, pz])
        momentum_sum_squared += np.dot(momentum, momentum)  # |p|^2
        sphericity_tensor += np.outer(momentum, momentum)  # p_i * p_j

    # Step 2: Normalize the sphericity tensor
    sphericity_tensor /= momentum_sum_squared

    # Step 3: Compute eigenvalues and sort them in descending order
    eigenvalues = np.linalg.eigvals(sphericity_tensor)
    eigenvalues = np.sort(eigenvalues)[::-1]  # λ1 >= λ2 >= λ3

    # Step 4: Calculate aplanarity (3/2 * smallest eigenvalue)
    aplanarity = 1.5 * eigenvalues[2]  # λ3

    return aplanarity


def compute_linear_aplanarity(vec1, vec2, vec3):
    """
    Computes the linear aplanarity for an event with exactly 3 TLorentzVector objects.
    
    Parameters:
        vec1, vec2, vec3: ROOT.TLorentzVector
            The 3 TLorentzVector objects representing particles in the event.
    
    Returns:
        float: The linear aplanarity value (0 ≤ A_linear ≤ 0.5).
    """
    # Step 1: Initialize sphericity tensor and total momentum sum squared
    sphericity_tensor = np.zeros((3, 3))
    momentum_sum_squared = 0.0

    # List of vectors
    particles = [vec1, vec2, vec3]

    for p in particles:
        px, py, pz = p.Px(), p.Py(), p.Pz()
        momentum = np.array([px, py, pz])
        momentum_sum_squared += np.dot(momentum, momentum)  # |p|^2
        sphericity_tensor += np.outer(momentum, momentum)  # p_i * p_j

    # Step 2: Normalize the sphericity tensor
    sphericity_tensor /= momentum_sum_squared

    # Step 3: Compute eigenvalues and sort them in descending order
    eigenvalues = np.linalg.eigvals(sphericity_tensor)
    eigenvalues = np.sort(eigenvalues)[::-1]  # λ1 >= λ2 >= λ3

    # Step 4: Linear aplanarity is the second eigenvalue
    linear_aplanarity = eigenvalues[1]  # λ2

    return linear_aplanarity

semileptonic_HL = GetRootFile(myFileName)
L_events,efficiencyL = PreSelectionCutEventsList(semileptonic_HL)
print("Efficiency preselection cut is: ", efficiencyL)

e4vecHL = electrons4Vec(semileptonic_HL,L_events)
jetLO4vecHL = jetLO4vec(semileptonic_HL,L_events)
jetNLO4vecHL = jetNLO4vec(semileptonic_HL,L_events)
met4VecHL = MET4Vec(semileptonic_HL,L_events)
HiggsRecMass_L = [ (e4vecHL[index] + jetLO4vecHL[index]+jetNLO4vecHL[index]).M()+met4VecHL[index].E() for index in range(len(e4vecHL)) ]

# Calculate statistics
mean_value = np.mean(HiggsRecMass_L)
std_dev = np.std(HiggsRecMass_L)
num_entries = len(HiggsRecMass_L)

# Create the histogram
plt.figure(figsize=(10.8, 9.6), dpi=300)
plt.hist(HiggsRecMass_L, bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.xlabel(r'Reconstructed Mass', fontsize=14)
plt.ylabel('Events frequency', fontsize=14)
plt.title(r'Histogram of Reconstructed Mass', fontsize=16)

# Add grid
#plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Add annotations for statistics
stats_text = (
    f"Mean: {mean_value:.2f}\n"
    f"Std Dev: {std_dev:.2f}\n"
    f"Entries: {num_entries}"
)
plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"))

# Improve tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adjust layout
plt.tight_layout()
plt.savefig("Muons Reconstructed_Mass.png")
# Show the plot

# DeltaR
DeltaR_JNLO_JLO_HL = ak.Array([jetNLO4vecHL[index].DeltaR(jetLO4vecHL[index]) for index in range(len(jetNLO4vecHL)) ])
DeltaR_electron_JLO_HL = ak.Array([e4vecHL[index].DeltaR(jetLO4vecHL[index]) for index in range(len(e4vecHL)) ])
DeltaR_electron_JNLO_HL = ak.Array([e4vecHL[index].DeltaR(jetNLO4vecHL[index]) for index in range(len(e4vecHL)) ])
# Delta Phi
DeltaPhi_electron_JLO_HL = ak.Array([e4vecHL[index].DeltaPhi(jetLO4vecHL[index]) for index in range(len(e4vecHL)) ])
DeltaPhi_electron_JNLO_HL = ak.Array([e4vecHL[index].DeltaPhi(jetNLO4vecHL[index]) for index in range(len(e4vecHL)) ])
DeltaPhi_JNLO_JLO_HL = ak.Array([jetNLO4vecHL[index].DeltaPhi(jetLO4vecHL[index]) for index in range(len(jetNLO4vecHL)) ])
## Sphericity JLO JNLO Electron
Sphericity_HL = ak.Array([compute_sphericity(e4vecHL[index], jetLO4vecHL[index], jetNLO4vecHL[index]) for index in range(len(e4vecHL))])
## Aplanarity JET LO JET NLO Electron
Aplanarity_HL = ak.Array([compute_aplanarity(e4vecHL[index], jetLO4vecHL[index], jetNLO4vecHL[index]) for index in range(len(e4vecHL))])
## Linear Aplanarity JET LO JET NLO Electron
Linear_Aplanarity_HL = ak.Array([compute_linear_aplanarity(e4vecHL[index], jetLO4vecHL[index], jetNLO4vecHL[index]) for index in range(len(e4vecHL))])
# Cos angle between two Objects j1l    j2l   j1j2
CosAngle_Electron_JLO_HL = ak.Array([np.cos(jetLO4vecHL[index].Angle(e4vecHL[index].Vect())) for index in range(len(jetLO4vecHL))]) # Cos (l, j LO)
CosAngle_Electron_JNLO_HL = ak.Array([np.cos(jetNLO4vecHL[index].Angle(e4vecHL[index].Vect())) for index in range(len(jetNLO4vecHL))]) # Cos (l, j LO)
CosAngle_JNLO_JLO_HL= ak.Array([np.cos(jetNLO4vecHL[index].Angle(jetLO4vecHL[index].Vect())) for index in range(len(jetNLO4vecHL))]) # Cos (j NLO, j LO)
# Cos theta objects themselves
cosTheta_JLO_HL = ak.Array([jetLO4vecHL[index].CosTheta() for index in range(len(jetLO4vecHL))])
cosTheta_JNLO_HL = ak.Array([jetNLO4vecHL[index].CosTheta() for index in range(len(jetNLO4vecHL))])
cosTheta_e_HL =ak.Array( [e4vecHL[index].CosTheta() for index in range(len(e4vecHL))])
# Theta Objects themselves
Theta_JLO_HL = ak.Array([jetLO4vecHL[index].Theta() for index in range(len(jetLO4vecHL))])
Theta_JNLO_HL = ak.Array([jetNLO4vecHL[index].Theta() for index in range(len(jetNLO4vecHL))])
Theta_e_HL =ak.Array( [e4vecHL[index].Theta() for index in range(len(e4vecHL))])
# Phi Objects themselves
Phi_JLO_HL = ak.Array([jetLO4vecHL[index].Phi() for index in range(len(jetLO4vecHL))])
Phi_JNLO_HL = ak.Array([jetNLO4vecHL[index].Phi() for index in range(len(jetNLO4vecHL))])
Phi_e_HL =ak.Array( [e4vecHL[index].Phi() for index in range(len(e4vecHL))])
# M()
M_JLO_HL = ak.Array([jetLO4vecHL[index].M() for index in range(len(jetLO4vecHL))])
M_JNLO_HL = ak.Array([jetNLO4vecHL[index].M() for index in range(len(jetNLO4vecHL))])
M_e_HL =ak.Array( [e4vecHL[index].M() for index in range(len(e4vecHL))])
M_met_HL =ak.Array( [met4VecHL[index].M() for index in range(len(met4VecHL))])
# Mt()
MT_JLO_HL = ak.Array([jetLO4vecHL[index].Mt() for index in range(len(jetLO4vecHL))])
MT_JNLO_HL = ak.Array([jetNLO4vecHL[index].Mt() for index in range(len(jetNLO4vecHL))])
MT_e_HL =ak.Array( [e4vecHL[index].Mt() for index in range(len(e4vecHL))])
MT_met_HL =ak.Array( [met4VecHL[index].Mt() for index in range(len(met4VecHL))])

# E()
Energy_JLO_HL = ak.Array([jetLO4vecHL[index].E() for index in range(len(jetLO4vecHL))])
Energy_JNLO_HL = ak.Array([jetNLO4vecHL[index].E() for index in range(len(jetNLO4vecHL))])
Energy_e_HL =ak.Array( [e4vecHL[index].E() for index in range(len(e4vecHL))])
Energy_met_HL =ak.Array( [met4VecHL[index].E() for index in range(len(met4VecHL))])

# Et()
Energy_Transverse_JLO_HL = ak.Array([jetLO4vecHL[index].Et() for index in range(len(jetLO4vecHL))])
Energy_Transverse_JNLO_HL = ak.Array([jetNLO4vecHL[index].Et() for index in range(len(jetNLO4vecHL))])
Energy_Transverse_e_HL =ak.Array( [e4vecHL[index].Et() for index in range(len(e4vecHL))])
Energy_Transverse_met_HL =ak.Array( [met4VecHL[index].Et() for index in range(len(met4VecHL))])

# Pt
PT_JLO_HL = ak.Array([jetLO4vecHL[index].Pt() for index in range(len(jetLO4vecHL))])
PT_JNLO_HL = ak.Array([jetNLO4vecHL[index].Pt() for index in range(len(jetNLO4vecHL))])
PT_e_HL =ak.Array( [e4vecHL[index].Pt() for index in range(len(e4vecHL))])
#Pseudorapidity Objects
MET_eta_HL = ak.Array([met4VecHL[index].PseudoRapidity() for index in range(len(met4VecHL))])# Pseudorapidity MET
electron_eta_HL = ak.Array([e4vecHL[index].PseudoRapidity() for index in range(len(e4vecHL))])# Pseudorapidity electron
JetLO_eta_HL = ak.Array([jetLO4vecHL[index].PseudoRapidity() for index in range(len(jetLO4vecHL))])# Pseudorapidity JLO
JetNLO_eta_HL = ak.Array([jetNLO4vecHL[index].PseudoRapidity() for index in range(len(jetNLO4vecHL))])# Pseudorapidity JNLO

#Delta Pseudorapidity 
Delta_eta_electron_JLO_HL = electron_eta_HL - JetLO_eta_HL
Delta_eta_electron_NJO_HL = electron_eta_HL - JetNLO_eta_HL
Delta_eta_JLO_JNLO_HL     = JetLO_eta_HL - JetNLO_eta_HL

#Rapidity Objects
MET_Rapidity_HL = ak.Array([met4VecHL[index].Rapidity() for index in range(len(met4VecHL))])# Pseudorapidity MET
electron_Rapidity_HL = ak.Array([e4vecHL[index].Rapidity() for index in range(len(e4vecHL))])# Pseudorapidity electron
JetLO_Rapidity_HL = ak.Array([jetLO4vecHL[index].Rapidity() for index in range(len(jetLO4vecHL))])# Pseudorapidity JLO
JetNLO_Rapidity_HL = ak.Array([jetNLO4vecHL[index].Rapidity() for index in range(len(jetNLO4vecHL))])# Pseudorapidity JNLO

# DeltaRapidity
Delta_rapidity_electron_JLO_HL = electron_Rapidity_HL - JetLO_Rapidity_HL
Delta_rapidity_electron_JNLO_HL= electron_Rapidity_HL - JetNLO_Rapidity_HL
Delta_rapidity_JNLO_JLO_HL = JetLO_Rapidity_HL - JetNLO_Rapidity_HL

# Create a ROOT file and write data to it
with uproot.recreate(myFileName+"_Muons") as file:
    file["events"] = {"Delta_rapidity_JNLO_JLO": Delta_rapidity_JNLO_JLO_HL,
                      "Delta_rapidity_Muon_JNLO" : Delta_rapidity_electron_JNLO_HL,
                      "Delta_rapidity_Muon_JLO" : Delta_rapidity_electron_JLO_HL,
                      "JetNLO_Rapidity" : JetNLO_Rapidity_HL,
                      "JetLO_Rapidity" : JetLO_Rapidity_HL,
                      "Muon_Rapidity" : electron_Rapidity_HL,
                      #"MET_Rapidity" : MET_Rapidity_HL,
                      "Delta_eta_JLO_JNLO" : Delta_eta_JLO_JNLO_HL,
                      "Delta_eta_Muon_NJO" : Delta_eta_electron_NJO_HL,
                      "Delta_eta_Muon_JLO" : Delta_eta_electron_JLO_HL,
                      "JetNLO_eta" : JetNLO_eta_HL,
                      "JetLO_eta" : JetLO_eta_HL,
                      "Muon_eta" : electron_eta_HL,
                      #"MET_eta" : MET_eta_HL,
                      "PT_Muon" : PT_e_HL,
                      "PT_JNLO" : PT_JNLO_HL,
                      "PT_JLO" : PT_JLO_HL,
                      "Energy_Transverse_Muon" : Energy_Transverse_e_HL,
                      "Energy_Transverse_JNLO" : Energy_Transverse_JNLO_HL,
                      "Energy_Transverse_JLO" : Energy_Transverse_JLO_HL,
                      "Energy_Transverse_met" : Energy_Transverse_met_HL,
                      "Energy_Muon" : Energy_e_HL,
                      "Energy_JNLO" : Energy_JNLO_HL,
                      "Energy_JLO" : Energy_JLO_HL,
                      "Energy_MET" : Energy_met_HL,
                      "M_JLO"   : M_JLO_HL,
                      "M_JNLO"  : M_JNLO_HL,
                      "M_Muon"     : M_e_HL,
                      "M_met"   :M_met_HL,
                      "MT_Muon" : MT_e_HL,
                      "MT_JNLO" : MT_JNLO_HL,
                      "MT_JLO" : MT_JLO_HL,
                      "MT_met" : MT_met_HL,
                      "Phi_Muon" : Phi_e_HL,
                      "Phi_JNLO" : Phi_JNLO_HL,
                      "Phi_JLO" : Phi_JLO_HL,
                      "Theta_Muon" : Theta_e_HL,
                      "Theta_JNLO" : Theta_JNLO_HL,
                      "Theta_JLO" : Theta_JLO_HL,
                      "cosTheta_Muon" : cosTheta_e_HL,
                      "cosTheta_JNLO" : cosTheta_JNLO_HL,
                      "cosTheta_JLO" : cosTheta_JLO_HL,
                      "CosAngle_JNLO_JLO" : CosAngle_JNLO_JLO_HL,
                      "CosAngle_Muon_JNLO" : CosAngle_Electron_JNLO_HL,
                      "CosAngle_Muon_JLO" : CosAngle_Electron_JLO_HL,
                      "Linear_Aplanarity" : Linear_Aplanarity_HL,
                      "Aplanarity" : Aplanarity_HL,
                      "Sphericity" : Sphericity_HL,
                      "DeltaPhi_JNLO_JLO" : DeltaPhi_JNLO_JLO_HL,
                      "DeltaPhi_Muon_JNLO" : DeltaPhi_electron_JNLO_HL,
                      "DeltaPhi_Muon_JLO" : DeltaPhi_electron_JLO_HL,
                      "DeltaR_JNLO_JLO" : DeltaR_JNLO_JLO_HL,
                      "DeltaR_Muon_JLO" : DeltaR_electron_JLO_HL,
                      "DeltaR_Muon_JNLO" :DeltaR_electron_JNLO_HL
                     }
    file.close()