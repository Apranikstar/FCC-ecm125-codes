import uproot
import numpy as np
import awkward as ak

def Sig(d23Value, d34Value, nIsoLepValue,nIsoPhoton, IMValue,METValue, pIsoLepValue,j1e,j2e):
    print("d23Value:",d23Value)
    print("d34Value:",d34Value)
    print("IMValue:",IMValue)
    print("METValue:",METValue)
    print("pIsoLepValue:",pIsoLepValue)
    print("j1e:",j1e)
    print("j2e:",j2e)

    def GetRootFile(rootFile):
        return uproot.open(rootFile)
    ###################################################################################################
    def PreSelectionCutEventsList(RootFile, d23Value, d34Value, nIsoLepValue,nIsoPhoton, IMValue,METValue, pIsoLepValue,j1e,j2e):
        e_jet1 = np.array(RootFile["events"]["jet1_e"].array())
        e_jet2 = np.array(RootFile["events"]["jet2_e"].array())

        n_isoleptons = np.array(RootFile["events"]["isoleptons_no"].array())
        n_isophotons = np.array(RootFile["events"]["isophotons_no"].array())
        IM = np.array(RootFile["events"]["IM"].array())
        MET_energy = np.array(RootFile["events"]["missingEnergy/missingEnergy.energy"].array())
    
        n_isoMuon = np.array(RootFile["events"]["isomuons_no"].array())
        p_isoMuon = RootFile["events"]["isomuons_p"].array()
        p_isoMuon =  np.array(ak.flatten(ak.where(ak.num(p_isoMuon) == 0, ak.Array([[0]] * len(p_isoMuon)), p_isoMuon)))

    
        n_isoElectron = np.array(RootFile["events"]["isoelectrons_no"].array())
        p_isoElectron = RootFile["events"]["isoelectrons_p"].array()
        p_isoElectron = np.array(ak.flatten(ak.where(ak.num(p_isoElectron) == 0, ak.Array([[0]] * len(p_isoElectron)), p_isoElectron)))

        d23 = np.array(RootFile["events"]["d23"].array())
        d34 = np.array(RootFile["events"]["d34"].array())

        eJet1Cut = np.where((j1e[0] < e_jet1) & (e_jet1 < j1e[1]) )[0]
        eJet2Cut = np.where((j2e[0] < e_jet2) & (e_jet2 < j2e[1]) )[0]

        d23Cut = np.where((d23Value[0] < d23) & (d23 < d23Value[1]) )[0]
        d34Cut = np.where((d34Value[0] < d34) & (d34 < d34Value[1]))[0]

        IsoLeptonsPreCut = np.where(n_isoleptons == nIsoLepValue)[0]
        IsoPhotonsPreCut = np.where(n_isophotons == nIsoPhoton)[0]

        IMPreCut = np.where((IMValue[0] < IM ) & (IM < IMValue[1]))[0]
    
        metPreCut = np.where((METValue[0] < MET_energy) & (MET_energy < METValue[1]))[0]

        p_isoMuonCut = np.where( (pIsoLepValue[0] < p_isoMuon)  & (p_isoMuon < pIsoLepValue[1]) )[0]
        p_isoElectronCut = np.where( (pIsoLepValue[0] < p_isoElectron) & (p_isoElectron <pIsoLepValue[1] ))[0]
        p_isoLeptonCut = np.concatenate((p_isoMuonCut, p_isoElectronCut))

        commonEvents = np.intersect1d(np.intersect1d(np.intersect1d(IsoLeptonsPreCut, IMPreCut), metPreCut),p_isoLeptonCut)
        commonEvents = np.intersect1d(np.intersect1d(np.intersect1d(commonEvents, d23Cut),d34Cut),IsoPhotonsPreCut)
        commonEvents = np.intersect1d(np.intersect1d(commonEvents,eJet1Cut),eJet2Cut)
        efficiency = np.divide(len(commonEvents) ,len(n_isoleptons))
        return commonEvents,efficiency
    ###################################################################################################    
    lum = 10000
    FileList = [
    'wzp6_ee_Hlnuqq_ecm125.root',
    'wzp6_ee_Hqqlnu_ecm125.root',
    'wzp6_ee_qq_ecm125.root',
    'wzp6_ee_eenunu_ecm125.root',
    'wzp6_ee_enueqq_ecm125.root',
    'wzp6_ee_Hgg_ecm125.root',
    'wzp6_ee_Hllnunu_ecm125.root',
    'wzp6_ee_l1l2nunu_ecm125.root',
    'wzp6_ee_mumununu_ecm125.root',
    'wzp6_ee_munumuqq_ecm125.root',
    'wzp6_ee_taunutauqq_ecm125.root',
    'wzp6_ee_tautaununu_ecm125.root'
    ]

    BkgList = {
    'wzp6_ee_qq_ecm125.root' : 363.1e3,
    'wzp6_ee_eenunu_ecm125.root':0.3364e3,
    'wzp6_ee_enueqq_ecm125.root':0.01382e3,
    'wzp6_ee_Hgg_ecm125.root':7.384e-2,
    'wzp6_ee_Hllnunu_ecm125.root':3.187e-2,
    'wzp6_ee_l1l2nunu_ecm125.root':0.005799e3,
    'wzp6_ee_mumununu_ecm125.root':0.2202e3,
    'wzp6_ee_munumuqq_ecm125.root':0.006711e3,
    'wzp6_ee_taunutauqq_ecm125.root':0.006761e3,
    'wzp6_ee_tautaununu_ecm125.root':0.04265e3
     }

    SgnlList ={
    'wzp6_ee_Hlnuqq_ecm125.root':4.584e-2,
    'wzp6_ee_Hqqlnu_ecm125.root':3.187e-2,
    }
    ###################################################################################################
    EfficiencyDict = {}
    for items in FileList:
        file = GetRootFile(items)
        EfficiencyDict[items] = PreSelectionCutEventsList(
                                                      file,
                                                      d23Value, d34Value,
                                                      nIsoLepValue,nIsoPhoton, IMValue, METValue, pIsoLepValue,
                                                      j1e,j2e
                                                      )
        print("Efficiency for ",items," is:", EfficiencyDict[items][1])
    ###################################################################################################
    
    nominatorHlnuqq = EfficiencyDict["wzp6_ee_Hlnuqq_ecm125.root"][1]*SgnlList["wzp6_ee_Hlnuqq_ecm125.root"] * lum
    nominatorHqqlnu = EfficiencyDict["wzp6_ee_Hqqlnu_ecm125.root"][1]*SgnlList["wzp6_ee_Hqqlnu_ecm125.root"] * lum
    denominator = []
    for items in BkgList.keys():
        denominator.append(EfficiencyDict[items][1] * BkgList[items] )
    significanceHlnuqq = nominatorHlnuqq / np.sqrt(sum(denominator)*lum)
    significanceHqqlnu = nominatorHqqlnu / np.sqrt(sum(denominator)*lum)
    print("###################################################################################################")
    print("Significance Hlnuqq =", significanceHlnuqq)
    print("Significance Hqqlnu =", significanceHqqlnu)
    print("###################################################################################################")
    ##EfficiencyDict["file.root"][0] = List of Selected Events, EfficiencyDict[["file.root"]][1] : Efficiency
    return significanceHlnuqq,significanceHqqlnu,EfficiencyDict

print("###################################################################################################")


Sig(d23Value=[0,35], d34Value=[0,10], nIsoLepValue=1,nIsoPhoton=0, IMValue=[75,85], METValue=[4,125], pIsoLepValue=[10,125], j1e=[40,60],j2e = [23,45])
Sig(d23Value=[0,35], d34Value=[0,10], nIsoLepValue=1,nIsoPhoton=0, IMValue=[75,85], METValue=[4,125], pIsoLepValue=[10,125], j1e=[40,60],j2e = [23,47])
Sig(d23Value=[0,35], d34Value=[0,10], nIsoLepValue=1,nIsoPhoton=0, IMValue=[75,85], METValue=[4,125], pIsoLepValue=[10,125], j1e=[40,60],j2e = [23,50])
Sig(d23Value=[0,35], d34Value=[0,10], nIsoLepValue=1,nIsoPhoton=0, IMValue=[75,85], METValue=[4,125], pIsoLepValue=[10,125], j1e=[40,60],j2e = [25,45])
Sig(d23Value=[0,35], d34Value=[0,10], nIsoLepValue=1,nIsoPhoton=0, IMValue=[75,85], METValue=[4,125], pIsoLepValue=[10,125], j1e=[40,60],j2e = [25,47])
Sig(d23Value=[0,35], d34Value=[0,10], nIsoLepValue=1,nIsoPhoton=0, IMValue=[75,85], METValue=[4,125], pIsoLepValue=[10,125], j1e=[40,60],j2e = [25,50])
Sig(d23Value=[0,35], d34Value=[0,10], nIsoLepValue=1,nIsoPhoton=0, IMValue=[75,85], METValue=[4,125], pIsoLepValue=[10,125], j1e=[40,60],j2e = [27,45])
Sig(d23Value=[0,35], d34Value=[0,10], nIsoLepValue=1,nIsoPhoton=0, IMValue=[75,85], METValue=[4,125], pIsoLepValue=[10,125], j1e=[40,60],j2e = [27,47])
Sig(d23Value=[0,35], d34Value=[0,10], nIsoLepValue=1,nIsoPhoton=0, IMValue=[75,85], METValue=[4,125], pIsoLepValue=[10,125], j1e=[40,60],j2e = [27,50])
