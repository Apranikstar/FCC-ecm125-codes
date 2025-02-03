/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides a simple example on how to use the trained classifiers
/// within an analysis module
/// - Project   : TMVA - a Root-integrated toolkit for multivariate data analysis
/// - Package   : TMVA
/// - Exectuable: TMVAClassificationApplication
///
/// \macro_output
/// \macro_code
/// \author Andreas Hoecker

#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"



using namespace TMVA;

void ElectronHlnuqq()// TString myMethodList = "" )
{

   double AS_cut[10][2]={0}; // number of events: AS_cut[n_samples][0=before cut, 1=after cut]
   //---------------------------------------------------------------
   // This loads the library
   TMVA::Tools::Instance();

   // Default MVA methods to be trained + tested
   std::map<std::string,int> Use;
   //============================================================= we have to swich on just 1 method in this application code:
   // Cut optimisation
   Use["Cuts"]            = 0;
   Use["CutsD"]           = 0;
   Use["CutsPCA"]         = 0;
   Use["CutsGA"]          = 0;
   Use["CutsSA"]          = 0;
   //
   // 1-dimensional likelihood ("naive Bayes estimator")
   Use["Likelihood"]      = 0;
   Use["LikelihoodD"]     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
   Use["LikelihoodMIX"]   = 0; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
   Use["LikelihoodKDE"]   = 0;
   Use["LikelihoodMIX"]   = 0;
   //
   // Mutidimensional likelihood and Nearest-Neighbour methods
   Use["PDERS"]           = 0;
   Use["PDERSD"]          = 0;
   Use["PDERSPCA"]        = 0;
   Use["PDEFoam"]         = 0;
   Use["PDEFoamBoost"]    = 0; // uses generalised MVA method boosting
   Use["KNN"]             = 0; // k-nearest neighbour method
   //
   // Linear Discriminant Analysis
   Use["LD"]              = 0; // Linear Discriminant identical to Fisher
   Use["Fisher"]          = 0;
   Use["FisherG"]         = 0;
   Use["BoostedFisher"]   = 0; // uses generalised MVA method boosting
   Use["HMatrix"]         = 0;
   //
   // Function Discriminant analysis
   Use["FDA_GA"]          = 0; // minimisation of user-defined function using Genetics Algorithm
   Use["FDA_SA"]          = 0;
   Use["FDA_MC"]          = 0;
   Use["FDA_MT"]          = 0;
   Use["FDA_GAMT"]        = 0;
   Use["FDA_MCMT"]        = 0;
   //
   // Neural Networks (all are feed-forward Multilayer Perceptrons)
   Use["MLP"]             = 0; // Recommended ANN
   Use["MLPBFGS"]         = 0; // Recommended ANN with optional training method
   Use["MLPBNN"]          = 0; // Recommended ANN with BFGS training method and bayesian regulator
   Use["CFMlpANN"]        = 0; // Depreciated ANN from ALEPH
   Use["TMlpANN"]         = 0; // ROOT's own ANN
   Use["DNN"]             = 0; // improved implementation of a NN (Deep Neural Network)
   //
   // Support Vector Machine
   Use["SVM"]             = 0;
   //
   // Boosted Decision Trees
   Use["BDT"]             = 1; // uses Adaptive Boost
   Use["BDTG"]            = 0; // uses Gradient Boost
   Use["BDTB"]            = 0; // uses Bagging
   Use["BDTD"]            = 0; // decorrelation + Adaptive Boost
   Use["BDTF"]            = 0; // allow usage of fisher discriminant for node splitting
   //
   // Friedman's RuleFit method, ie, an optimised series of cuts ("rules")
   Use["RuleFit"]         = 0;
   // ---------------------------------------------------------------
   Use["Plugin"]          = 0;
   Use["Category"]        = 0;
   Use["SVM_Gauss"]       = 0;
   Use["SVM_Poly"]        = 0;
   Use["SVM_Lin"]         = 0;


   std::cout << std::endl;
   std::cout << "==> Start TMVAClassificationApplicationFCCee" << std::endl;

   // Select methods (don't look at this code - not of interest)
   if (myMethodList != "") {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

      std::vector<TString> mlist = gTools().SplitString( myMethodList, ',' );
      for (UInt_t i=0; i<mlist.size(); i++) {
         std::string regMethod(mlist[i]);

         if (Use.find(regMethod) == Use.end()) {
            std::cout << "Method \"" << regMethod
                      << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
            for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
               std::cout << it->first << " ";
            }
            std::cout << std::endl;
            return;
         }
         Use[regMethod] = 1;
      }
   }

   // --------------------------------------------------------------------------------------------------

   // Create the Reader object

   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );

   // Create a set of variables and declare them to the reader
   // - the variable names MUST corresponds in name and type to those given in the weight file(s) used

   // ======================================================================================================

    Float_t Delta_rapidity_JNLO_JLO;
    Float_t Delta_rapidity_electron_JNLO;
    Float_t Delta_rapidity_electron_JLO;
    Float_t JetNLO_Rapidity;
    Float_t JetLO_Rapidity;
    Float_t electron_Rapidity;
    Float_t Delta_eta_JLO_JNLO;
    Float_t Delta_eta_electron_NJO;
    Float_t Delta_eta_electron_JLO;
    Float_t JetNLO_eta;
    Float_t JetLO_eta;
    Float_t electron_eta;
    Float_t PT_electron;
    Float_t PT_JNLO;
    Float_t PT_JLO;
    Float_t Energy_Transverse_electron;
    Float_t Energy_Transverse_JNLO;
    Float_t Energy_Transverse_JLO;
    Float_t Energy_Transverse_met;
    Float_t Energy_electron;
    Float_t Energy_JNLO;
    Float_t Energy_JLO;
    Float_t M_JLO;
    Float_t M_JNLO;
    Float_t M_electron;
    Float_t M_met;
    Float_t MT_electron;
    Float_t MT_JNLO;
    Float_t MT_JLO;
    Float_t MT_met;
    Float_t Phi_electron;
    Float_t Phi_JNLO;
    Float_t Phi_JLO;
    Float_t Theta_electron;
    Float_t Theta_JNLO;
    Float_t Theta_JLO;
    Float_t cosTheta_electron;
    Float_t cosTheta_JNLO;
    Float_t cosTheta_JLO;
    Float_t CosAngle_JNLO_JLO;
    Float_t CosAngle_Electron_JNLO;
    Float_t CosAngle_Electron_JLO;
    Float_t Linear_Aplanarity;
    Float_t Aplanarity;
    Float_t Sphericity;
    Float_t DeltaPhi_JNLO_JLO;
    Float_t DeltaPhi_electron_JNLO;
    Float_t DeltaPhi_electron_JLO;
    Float_t DeltaR_JNLO_JLO;
    Float_t DeltaR_electron_JLO;
    Float_t DeltaR_electron_JNLO;
    Float_t Energy_MET;


// ======================================================================================================
    reader->AddVariable("Delta_rapidity_JNLO_JLO", &Delta_rapidity_JNLO_JLO);
    reader->AddVariable("Delta_rapidity_electron_JNLO", &Delta_rapidity_electron_JNLO);
    reader->AddVariable("Delta_rapidity_electron_JLO", &Delta_rapidity_electron_JLO);
    reader->AddVariable("JetNLO_Rapidity", &JetNLO_Rapidity);
    reader->AddVariable("JetLO_Rapidity", &JetLO_Rapidity);
    reader->AddVariable("electron_Rapidity", &electron_Rapidity);
    reader->AddVariable("Delta_eta_JLO_JNLO", &Delta_eta_JLO_JNLO);
    reader->AddVariable("Delta_eta_electron_NJO", &Delta_eta_electron_NJO);
    reader->AddVariable("Delta_eta_electron_JLO", &Delta_eta_electron_JLO);
    reader->AddVariable("JetNLO_eta", &JetNLO_eta);
    reader->AddVariable("JetLO_eta", &JetLO_eta);
    reader->AddVariable("electron_eta", &electron_eta);
    reader->AddVariable("PT_electron", &PT_electron);
    reader->AddVariable("PT_JNLO", &PT_JNLO);
    reader->AddVariable("PT_JLO", &PT_JLO);
    reader->AddVariable("Energy_Transverse_electron", &Energy_Transverse_electron);
    reader->AddVariable("Energy_Transverse_JNLO", &Energy_Transverse_JNLO);
    reader->AddVariable("Energy_Transverse_JLO", &Energy_Transverse_JLO);
    reader->AddVariable("Energy_Transverse_met", &Energy_Transverse_met);
    reader->AddVariable("Energy_electron", &Energy_electron);
    reader->AddVariable("Energy_JNLO", &Energy_JNLO);
    reader->AddVariable("Energy_JLO", &Energy_JLO);
    reader->AddVariable("M_JLO", &M_JLO);
    reader->AddVariable("M_JNLO", &M_JNLO);
    reader->AddVariable("M_electron", &M_electron);
    reader->AddVariable("M_met", &M_met);
    reader->AddVariable("MT_electron", &MT_electron);
    reader->AddVariable("MT_JNLO", &MT_JNLO);
    reader->AddVariable("MT_JLO", &MT_JLO);
    reader->AddVariable("MT_met", &MT_met);
    reader->AddVariable("Phi_electron", &Phi_electron);
    reader->AddVariable("Phi_JNLO", &Phi_JNLO);
    reader->AddVariable("Phi_JLO", &Phi_JLO);
    reader->AddVariable("Theta_electron", &Theta_electron);
    reader->AddVariable("Theta_JNLO", &Theta_JNLO);
    reader->AddVariable("Theta_JLO", &Theta_JLO);
    reader->AddVariable("cosTheta_electron", &cosTheta_electron);
    reader->AddVariable("cosTheta_JNLO", &cosTheta_JNLO);
    reader->AddVariable("cosTheta_JLO", &cosTheta_JLO);
    reader->AddVariable("CosAngle_JNLO_JLO", &CosAngle_JNLO_JLO);
    reader->AddVariable("CosAngle_Electron_JNLO", &CosAngle_Electron_JNLO);
    reader->AddVariable("CosAngle_Electron_JLO", &CosAngle_Electron_JLO);
    reader->AddVariable("Linear_Aplanarity", &Linear_Aplanarity);
    reader->AddVariable("Aplanarity", &Aplanarity);
    reader->AddVariable("Sphericity", &Sphericity);
    reader->AddVariable("DeltaPhi_JNLO_JLO", &DeltaPhi_JNLO_JLO);
    reader->AddVariable("DeltaPhi_electron_JNLO", &DeltaPhi_electron_JNLO);
    reader->AddVariable("DeltaPhi_electron_JLO", &DeltaPhi_electron_JLO);
    reader->AddVariable("DeltaR_JNLO_JLO", &DeltaR_JNLO_JLO);
    reader->AddVariable("DeltaR_electron_JLO", &DeltaR_electron_JLO);
    reader->AddVariable("DeltaR_electron_JNLO", &DeltaR_electron_JNLO);
    reader->AddVariable("Energy_MET", &Energy_MET);


// ======================================================================================================

   // Book the MVA methods

   TString dir    = "Electron_DataSet/weights/";
   TString prefix = "TMVA_Classification";

   // Book method(s)
   for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
      if (it->second) {
         TString methodName = TString(it->first) + TString(" method");
         TString weightfile = dir + prefix + TString("_") + TString(it->first) + TString(".weights.xml");
         reader->BookMVA( methodName, weightfile );
      }
   }
   
     // Book output histograms

   UInt_t nbin = 100;
/*
   TH1F * histBdt_ctll;
   TH1F * histBdt_ctrr;
   TH1F * histBdt_ctrl;
   TH1F * histBdt_ctlr;
   TH1F * histBdt_bkg_etavv;
   TH1F * histBdt_bkg_llqq;
   TH1F * histBdt_bkg_tata;
   TH1F * histBdt_bkg_llll;
*/

   TH1F * histBdtG[8];

 // *histBdtG[3], *histBdtD[3];
   //TFile * input[3];
   //TString fname[3];
  
/*
   if (Use["BDT"])     histBdt_ctll      = new TH1F( "signal", "signal", nbin, -1.0, 1.0 );

  if (Use["BDT"])     histBdt_bkg_etavv      = new TH1F( "etavv",  "etavv", nbin, -1.0, 1.0 );
   if (Use["BDT"])     histBdt_bkg_llqq      = new TH1F( "llqq",  "llqq", nbin, -1.0, 1.0 );
   if (Use["BDT"])     histBdt_bkg_tata      = new TH1F( "tata",  "tata", nbin, -1.0, 1.0 );
   if (Use["BDT"])     histBdt_bkg_llll      = new TH1F( "llll",  "llll", nbin, -1.0, 1.0 );
*/

   if (Use["BDT"])
    {
     histBdtG[0]      = new TH1F( "Hlnuqq", "Hlnuqq", nbin, -1.0, 1.0 );
     histBdtG[1]      = new TH1F( "eenunuFile",  "eenunuFile", nbin, -1.0, 1.0 );
     histBdtG[2]      = new TH1F( "enueqqFile",  "enueqqFile", nbin, -1.0, 1.0 );
     histBdtG[3]      = new TH1F( "munumuqqFile",  "munumuqqFile", nbin, -1.0, 1.0 );
     histBdtG[4]      = new TH1F( "tautaununuFile",  "tautaununuFile", nbin, -1.0, 1.0 );
     histBdtG[5]      = new TH1F( "taunutauqqFile",  "taunutauqqFile", nbin, -1.0, 1.0 );
     histBdtG[6]      = new TH1F( "l1l2nunuFile",  "l1l2nunuFile", nbin, -1.0, 1.0 );
     histBdtG[7]      = new TH1F( "qqFile",  "qqFile", nbin, -1.0, 1.0 );
     histBdtG[8]      = new TH1F( "HllnunuFile",  "HllnunuFile", nbin, -1.0, 1.0 );
     histBdtG[9]      = new TH1F( "HggFile",  "HggFile", nbin, -1.0, 1.0 );


    }

/*
****** we have to define these for each sample. We can use array like: histLk[samples]

   if (Use["Likelihood"])    histLk      = new TH1F( "MVA_Likelihood",    "MVA_Likelihood",    nbin, -1, 1 );
   if (Use["LikelihoodD"])   histLkD     = new TH1F( "MVA_LikelihoodD",   "MVA_LikelihoodD",   nbin, -1, 0.9999 );
   if (Use["LikelihoodMIX"]) histLkPCA   = new TH1F( "MVA_LikelihoodMIX", "MVA_LikelihoodMIX", nbin, -1, 1 );
   if (Use["LikelihoodKDE"]) histLkKDE   = new TH1F( "MVA_LikelihoodKDE", "MVA_LikelihoodKDE", nbin,  -0.00001, 0.99999 );
   if (Use["LikelihoodMIX"]) histLkMIX   = new TH1F( "MVA_LikelihoodMIX", "MVA_LikelihoodMIX", nbin,  0, 1 );
   if (Use["PDERS"])         histPD      = new TH1F( "MVA_PDERS",         "MVA_PDERS",         nbin,  0, 1 );
   if (Use["PDERSD"])        histPDD     = new TH1F( "MVA_PDERSD",        "MVA_PDERSD",        nbin,  0, 1 );
   if (Use["PDERSPCA"])      histPDPCA   = new TH1F( "MVA_PDERSPCA",      "MVA_PDERSPCA",      nbin,  0, 1 );
   if (Use["KNN"])           histKNN     = new TH1F( "MVA_KNN",           "MVA_KNN",           nbin,  0, 1 );
   if (Use["HMatrix"])       histHm      = new TH1F( "MVA_HMatrix",       "MVA_HMatrix",       nbin, -0.95, 1.55 );
   if (Use["Fisher"])        histFi      = new TH1F( "MVA_Fisher",        "MVA_Fisher",        nbin, -4, 4 );
   if (Use["FisherG"])       histFiG     = new TH1F( "MVA_FisherG",       "MVA_FisherG",       nbin, -1, 1 );
   if (Use["BoostedFisher"]) histFiB     = new TH1F( "MVA_BoostedFisher", "MVA_BoostedFisher", nbin, -2, 2 );
   if (Use["LD"])            histLD      = new TH1F( "MVA_LD",            "MVA_LD",            nbin, -2, 2 );
   if (Use["MLP"])           histNn      = new TH1F( "MVA_MLP",           "MVA_MLP",           nbin, -1.25, 1.5 );
   if (Use["MLPBFGS"])       histNnbfgs  = new TH1F( "MVA_MLPBFGS",       "MVA_MLPBFGS",       nbin, -1.25, 1.5 );
   if (Use["MLPBNN"])        histNnbnn   = new TH1F( "MVA_MLPBNN",        "MVA_MLPBNN",        nbin, -1.25, 1.5 );
   if (Use["CFMlpANN"])      histNnC     = new TH1F( "MVA_CFMlpANN",      "MVA_CFMlpANN",      nbin,  0, 1 );
   if (Use["TMlpANN"])       histNnT     = new TH1F( "MVA_TMlpANN",       "MVA_TMlpANN",       nbin, -1.3, 1.3 );
   if (Use["DNN"])           histNdn     = new TH1F( "MVA_DNN",           "MVA_DNN",           nbin, -0.1, 1.1 );
   if (Use["BDT"])           histBdt     = new TH1F( "MVA_BDT",           "MVA_BDT",           nbin, -1.0, 1.0 );
   if (Use["BDTG"])          histBdtG    = new TH1F( "MVA_BDTG",          "MVA_BDTG",          nbin, -1.0, 1.0 );
   if (Use["BDTB"])          histBdtB    = new TH1F( "MVA_BDTB",          "MVA_BDTB",          nbin, -1.0, 1.0 );
   if (Use["BDTD"])          histBdtD    = new TH1F( "MVA_BDTD",          "MVA_BDTD",          nbin, -1.0, 1.0 );
   if (Use["BDTF"])          histBdtF    = new TH1F( "MVA_BDTF",          "MVA_BDTF",          nbin, -1.0, 1.0 );
   if (Use["RuleFit"])       histRf      = new TH1F( "MVA_RuleFit",       "MVA_RuleFit",       nbin, -2.0, 2.0 );
   if (Use["SVM_Gauss"])     histSVMG    = new TH1F( "MVA_SVM_Gauss",     "MVA_SVM_Gauss",     nbin,  0.0, 1.0 );
   if (Use["SVM_Poly"])      histSVMP    = new TH1F( "MVA_SVM_Poly",      "MVA_SVM_Poly",      nbin,  0.0, 1.0 );
   if (Use["SVM_Lin"])       histSVML    = new TH1F( "MVA_SVM_Lin",       "MVA_SVM_Lin",       nbin,  0.0, 1.0 );
   if (Use["FDA_MT"])        histFDAMT   = new TH1F( "MVA_FDA_MT",        "MVA_FDA_MT",        nbin, -2.0, 3.0 );
   if (Use["FDA_GA"])        histFDAGA   = new TH1F( "MVA_FDA_GA",        "MVA_FDA_GA",        nbin, -2.0, 3.0 );
   if (Use["Category"])      histCat     = new TH1F( "MVA_Category",      "MVA_Category",      nbin, -2., 2. );
   if (Use["Plugin"])        histPBdt    = new TH1F( "MVA_PBDT",          "MVA_BDT",           nbin, -1.0, 1.0 );
*/

   //for ( int ifile = 1; ifile < 3; ifile++ ) { //ifile = sample numbers

   //if (Use["BDT"])     histBdt[ifile]      = new TH1F( "MVA_BDT",           "MVA_BDT",           nbin, -1.0, 1.0 );
   //if (Use["BDTD"])    histBdtD[ifile]     = new TH1F( "TMVA_BDTD",         "TMVA_BDTD",         nbin, -0.6, 0.6 );
   //if (Use["BDTG"])    histBdtG[ifile]     = new TH1F( "TMVA_BDTG",         "TMVA_BDTG",         nbin, -1.0, 1.0 );


   // Prepare input tree (this must be replaced by your data source)
   // in this example, there is a toy tree with signal and one with background events
   // we'll later on use only the "signal" events for the test in this example.
   //

//======================================================================================================
   TFile *input(0);

      //input = TFile::Open( "region1TMVA_input13TeV.root" ); // check if file in local directory exists
      //input2 = TFile::Open( "region1TMVA_input13TeV.root" ); // check if file in local directory exists
    HlnuqqFile = TFile::Open("Electron_DataBase/Electron_wzp6_ee_Hlnuqq_ecm125.root")

    eenunuFile = TFile::Open("Electron_DataBase/Electron_wzp6_ee_eenunu_ecm125.root")

    enueqqFile = TFile::Open("Electron_DataBase/Electron_wzp6_ee_enueqq_ecm125.root")

    munumuqqFile = TFile::Open("Electron_DataBase/Electron_wzp6_ee_munumuqq_ecm125.root")

    taunutauqqFile = TFile::Open("Electron_DataBase/Electron_wzp6_ee_taunutauqq_ecm125.root")

    tautaununuFile = TFile::Open("Electron_DataBase/Electron_wzp6_ee_tautaununu_ecm125.root")

    l1l2nunuFile = TFile::Open("Electron_DataBase/Electron_wzp6_ee_l1l2nunu_ecm125.root")

    qqFile = TFile::Open("Electron_DataBase/Electron_wzp6_ee_qq_ecm125.root")

    HllnunuFile = TFile::Open("Electron_DataBase/Electron_wzp6_ee_Hllnunu_ecm125.root")

    HggFile = TFile::Open("Electron_DataBase/Electron_wzp6_ee_Hgg_ecm125.root")

   std::cout << "--- TMVAClassificationApp    : Using input file: " << std::endl;

   // Event loop

   // Prepare the event tree
   // - Here the variable names have to corresponds to your tree
   // - You can use the same variables as above which is slightly faster,
   //   but of course you can use different ones and copy the values inside the event loop

   std::cout << "--- Select signal sample" << std::endl;
   TTree *theTree[10];

   theTree[0] = (TTree*)HlnuqqFile->Get("events");
   theTree[1] =  (TTree*)eenunuFile->Get("events");
   theTree[2] =  (TTree*)enueqqFile->Get("events");
   theTree[3] =  (TTree*)munumuqqFile->Get("events");
   theTree[4] =  (TTree*)taunutauqqFile->Get("events");
   theTree[5] =  (TTree*)tautaununuFile->Get("events");
   theTree[6] =  (TTree*)l1l2nunuFile->Get("events");
   theTree[7] =  (TTree*)qqFile->Get("events");
   theTree[8] =  (TTree*)HllnunuFile->Get("events");
   theTree[9] =  (TTree*)HggFile->Get("events");


   Double_t weight [10] = {6.78E-17, 5.6e-14, 9.98e-14, 1.58e-15, 9.33e-15, 1.56e-14, 2.23e-14, 2.95e-10, 7.48e-17, 8.1e-18             };// = sigam_MG*eff*lum

    const Double_t sigma_MG [10] = {4.58E-05, 0.3364, 0.01382, 0.006711, 0.006761,   0.04265,      0.005799, 363.1, 3.19E-05 , 7.38E-05}; // (pb)
    //t_br = 0.1359855, t_bi = 0.1188306 [pb]

    const Double_t eff_pre [10] = {0.268435833 , 0.018510909, 0.649714444, 0.023485, 0.124132222, 0.036679, 0.461105833, 0.06498625 , 0.281606667 , 0.013171667}; //eff of preselection cuts,
    // t_br_eff_pre =0 .144391, t_bi_eff_pre = 0.144394
    
    
    
    /*
 
   Double_t Weight_t_br     = weight[0];  for t_br = 19635.1
   Double_t Weight_t_bi     = weight[1];  for t_bi = 17158.5
   Double_t Weight_ep2bbjvbkg1 = weight[2];
   Double_t Weight_ep2ccjvbkg2 = weight[3];
   Double_t Weight_ep2vjz2bbbkg3 = weight[4];
   Double_t Weight_ep2vjz2ccbkg4 = weight[5];
   Double_t Weight_ep2vjz2jjbkg5 = weight[6];
   Double_t Weight_ep2jt2w2jjbkg6 = weight[7];
   Double_t Weight_ep2vjhSMbkg7 = weight[8];
*/

  // Double_t user_ljet_eta, user_bjetsInvariantMass, user_MET, user_HT, user_deltaR_bjets, user_deltaR_ljetbjet, user_deltaR_ljetsubbjet, user_cos_bjets, user_alphaZMF;//due to "double v.s. float" Error , user_variable;
    double_t user_Delta_rapidity_JNLO_JLO;
    double_t user_Delta_rapidity_electron_JNLO;
    double_t user_Delta_rapidity_electron_JLO;
    double_t user_JetNLO_Rapidity;
    double_t user_JetLO_Rapidity;
    double_t user_electron_Rapidity;
    double_t user_Delta_eta_JLO_JNLO;
    double_t user_Delta_eta_electron_NJO;
    double_t user_Delta_eta_electron_JLO;
    double_t user_JetNLO_eta;
    double_t user_JetLO_eta;
    double_t user_electron_eta;
    double_t user_PT_electron;
    double_t user_PT_JNLO;
    double_t user_PT_JLO;
    double_t user_Energy_Transverse_electron;
    double_t user_Energy_Transverse_JNLO;
    double_t user_Energy_Transverse_JLO;
    double_t user_Energy_Transverse_met;
    double_t user_Energy_electron;
    double_t user_Energy_JNLO;
    double_t user_Energy_JLO;
    double_t user_M_JLO;
    double_t user_M_JNLO;
    double_t user_M_electron;
    double_t user_M_met;
    double_t user_MT_electron;
    double_t user_MT_JNLO;
    double_t user_MT_JLO;
    double_t user_MT_met;
    double_t user_Phi_electron;
    double_t user_Phi_JNLO;
    double_t user_Phi_JLO;
    double_t user_Theta_electron;
    double_t user_Theta_JNLO;
    double_t user_Theta_JLO;
    double_t user_cosTheta_electron;
    double_t user_cosTheta_JNLO;
    double_t user_cosTheta_JLO;
    double_t user_CosAngle_JNLO_JLO;
    double_t user_CosAngle_Electron_JNLO;
    double_t user_CosAngle_Electron_JLO;
    double_t user_Linear_Aplanarity;
    double_t user_Aplanarity;
    double_t user_Sphericity;
    double_t user_DeltaPhi_JNLO_JLO;
    double_t user_DeltaPhi_electron_JNLO;
    double_t user_DeltaPhi_electron_JLO;
    double_t user_DeltaR_JNLO_JLO;
    double_t user_DeltaR_electron_JLO;
    double_t user_DeltaR_electron_JNLO;
    double_t user_Energy_MET;



// ======================================================================================================== 
   for (int ii = 0; ii < 10; ii++)
   { // sample loop

       theTree[ii]->SetBranchAddress("Delta_rapidity_JNLO_JLO", &user_Delta_rapidity_JNLO_JLO);
       theTree[ii]->SetBranchAddress("Delta_rapidity_electron_JNLO", &user_Delta_rapidity_electron_JNLO);
       theTree[ii]->SetBranchAddress("Delta_rapidity_electron_JLO", &user_Delta_rapidity_electron_JLO);
       theTree[ii]->SetBranchAddress("JetNLO_Rapidity", &user_JetNLO_Rapidity);
       theTree[ii]->SetBranchAddress("JetLO_Rapidity", &user_JetLO_Rapidity);
       theTree[ii]->SetBranchAddress("electron_Rapidity", &user_electron_Rapidity);
       theTree[ii]->SetBranchAddress("Delta_eta_JLO_JNLO", &user_Delta_eta_JLO_JNLO);
       theTree[ii]->SetBranchAddress("Delta_eta_electron_NJO", &user_Delta_eta_electron_NJO);
       theTree[ii]->SetBranchAddress("Delta_eta_electron_JLO", &user_Delta_eta_electron_JLO);
       theTree[ii]->SetBranchAddress("JetNLO_eta", &user_JetNLO_eta);
       theTree[ii]->SetBranchAddress("JetLO_eta", &user_JetLO_eta);
       theTree[ii]->SetBranchAddress("electron_eta", &user_electron_eta);
       theTree[ii]->SetBranchAddress("PT_electron", &user_PT_electron);
       theTree[ii]->SetBranchAddress("PT_JNLO", &user_PT_JNLO);
       theTree[ii]->SetBranchAddress("PT_JLO", &user_PT_JLO);
       theTree[ii]->SetBranchAddress("Energy_Transverse_electron", &user_Energy_Transverse_electron);
       theTree[ii]->SetBranchAddress("Energy_Transverse_JNLO", &user_Energy_Transverse_JNLO);
       theTree[ii]->SetBranchAddress("Energy_Transverse_JLO", &user_Energy_Transverse_JLO);
       theTree[ii]->SetBranchAddress("Energy_Transverse_met", &user_Energy_Transverse_met);
       theTree[ii]->SetBranchAddress("Energy_electron", &user_Energy_electron);
       theTree[ii]->SetBranchAddress("Energy_JNLO", &user_Energy_JNLO);
       theTree[ii]->SetBranchAddress("Energy_JLO", &user_Energy_JLO);
       theTree[ii]->SetBranchAddress("M_JLO", &user_M_JLO);
       theTree[ii]->SetBranchAddress("M_JNLO", &user_M_JNLO);
       theTree[ii]->SetBranchAddress("M_electron", &user_M_electron);
       theTree[ii]->SetBranchAddress("M_met", &user_M_met);
       theTree[ii]->SetBranchAddress("MT_electron", &user_MT_electron);
       theTree[ii]->SetBranchAddress("MT_JNLO", &user_MT_JNLO);
       theTree[ii]->SetBranchAddress("MT_JLO", &user_MT_JLO);
       theTree[ii]->SetBranchAddress("MT_met", &user_MT_met);
       theTree[ii]->SetBranchAddress("Phi_electron", &user_Phi_electron);
       theTree[ii]->SetBranchAddress("Phi_JNLO", &user_Phi_JNLO);
       theTree[ii]->SetBranchAddress("Phi_JLO", &user_Phi_JLO);
       theTree[ii]->SetBranchAddress("Theta_electron", &user_Theta_electron);
       theTree[ii]->SetBranchAddress("Theta_JNLO", &user_Theta_JNLO);
       theTree[ii]->SetBranchAddress("Theta_JLO", &user_Theta_JLO);
       theTree[ii]->SetBranchAddress("cosTheta_electron", &user_cosTheta_electron);
       theTree[ii]->SetBranchAddress("cosTheta_JNLO", &user_cosTheta_JNLO);
       theTree[ii]->SetBranchAddress("cosTheta_JLO", &user_cosTheta_JLO);
       theTree[ii]->SetBranchAddress("CosAngle_JNLO_JLO", &user_CosAngle_JNLO_JLO);
       theTree[ii]->SetBranchAddress("CosAngle_Electron_JNLO", &user_CosAngle_Electron_JNLO);
       theTree[ii]->SetBranchAddress("CosAngle_Electron_JLO", &user_CosAngle_Electron_JLO);
       theTree[ii]->SetBranchAddress("Linear_Aplanarity", &user_Linear_Aplanarity);
       theTree[ii]->SetBranchAddress("Aplanarity", &user_Aplanarity);
       theTree[ii]->SetBranchAddress("Sphericity", &user_Sphericity);
       theTree[ii]->SetBranchAddress("DeltaPhi_JNLO_JLO", &user_DeltaPhi_JNLO_JLO);
       theTree[ii]->SetBranchAddress("DeltaPhi_electron_JNLO", &user_DeltaPhi_electron_JNLO);
       theTree[ii]->SetBranchAddress("DeltaPhi_electron_JLO", &user_DeltaPhi_electron_JLO);
       theTree[ii]->SetBranchAddress("DeltaR_JNLO_JLO", &user_DeltaR_JNLO_JLO);
       theTree[ii]->SetBranchAddress("DeltaR_electron_JLO", &user_DeltaR_electron_JLO);
       theTree[ii]->SetBranchAddress("DeltaR_electron_JNLO", &user_DeltaR_electron_JNLO);
       theTree[ii]->SetBranchAddress("Energy_MET", &user_Energy_MET);   


//============================================================================================================
    TStopwatch sw;
    sw.Start();

//----------------------------------------------------------------------------------------

    
        // for (int ievt0=0; ievt0<10000;ievt0++) {  // Event loop
    for ( int ievt0 = 0; ievt0 < theTree[ii]-> GetEntries(); ievt0++ ) {  // Event loop

       //if (ievt0%10000 == 0) std::cout << "--- ... Processing event: " << ievt0 << std::endl;

          AS_cut[ii][0]++;
          theTree[ii] -> GetEntry(ievt0);  
      
          //jet_pt= float(user_jet_pt);
        Delta_rapidity_JNLO_JLO = float(user_Delta_rapidity_JNLO_JLO);
        Delta_rapidity_electron_JNLO = float(user_Delta_rapidity_electron_JNLO);
        Delta_rapidity_electron_JLO = float(user_Delta_rapidity_electron_JLO);
        JetNLO_Rapidity = float(user_JetNLO_Rapidity);
        JetLO_Rapidity = float(user_JetLO_Rapidity);
        electron_Rapidity = float(user_electron_Rapidity);
        Delta_eta_JLO_JNLO = float(user_Delta_eta_JLO_JNLO);
        Delta_eta_electron_NJO = float(user_Delta_eta_electron_NJO);
        Delta_eta_electron_JLO = float(user_Delta_eta_electron_JLO);
        JetNLO_eta = float(user_JetNLO_eta);
        JetLO_eta = float(user_JetLO_eta);
        electron_eta = float(user_electron_eta);
        PT_electron = float(user_PT_electron);
        PT_JNLO = float(user_PT_JNLO);
        PT_JLO = float(user_PT_JLO);
        Energy_Transverse_electron = float(user_Energy_Transverse_electron);
        Energy_Transverse_JNLO = float(user_Energy_Transverse_JNLO);
        Energy_Transverse_JLO = float(user_Energy_Transverse_JLO);
        Energy_Transverse_met = float(user_Energy_Transverse_met);
        Energy_electron = float(user_Energy_electron);
        Energy_JNLO = float(user_Energy_JNLO);
        Energy_JLO = float(user_Energy_JLO);
        M_JLO = float(user_M_JLO);
        M_JNLO = float(user_M_JNLO);
        M_electron = float(user_M_electron);
        M_met = float(user_M_met);
        MT_electron = float(user_MT_electron);
        MT_JNLO = float(user_MT_JNLO);
        MT_JLO = float(user_MT_JLO);
        MT_met = float(user_MT_met);
        Phi_electron = float(user_Phi_electron);
        Phi_JNLO = float(user_Phi_JNLO);
        Phi_JLO = float(user_Phi_JLO);
        Theta_electron = float(user_Theta_electron);
        Theta_JNLO = float(user_Theta_JNLO);
        Theta_JLO = float(user_Theta_JLO);
        cosTheta_electron = float(user_cosTheta_electron);
        cosTheta_JNLO = float(user_cosTheta_JNLO);
        cosTheta_JLO = float(user_cosTheta_JLO);
        CosAngle_JNLO_JLO = float(user_CosAngle_JNLO_JLO);
        CosAngle_Electron_JNLO = float(user_CosAngle_Electron_JNLO);
        CosAngle_Electron_JLO = float(user_CosAngle_Electron_JLO);
        Linear_Aplanarity = float(user_Linear_Aplanarity);
        Aplanarity = float(user_Aplanarity);
        Sphericity = float(user_Sphericity);
        DeltaPhi_JNLO_JLO = float(user_DeltaPhi_JNLO_JLO);
        DeltaPhi_electron_JNLO = float(user_DeltaPhi_electron_JNLO);
        DeltaPhi_electron_JLO = float(user_DeltaPhi_electron_JLO);
        DeltaR_JNLO_JLO = float(user_DeltaR_JNLO_JLO);
        DeltaR_electron_JLO = float(user_DeltaR_electron_JLO);
        DeltaR_electron_JNLO = float(user_DeltaR_electron_JNLO);
        Energy_MET = float(user_Energy_MET);




      }

//-------------------------------------------------------------------
      if(Use["BDT"])   histBdtG[ii]->Fill( reader->EvaluateMVA("BDT method"), weight[ii] ); //change the weight number in the weight array for t_bi

      if(Use["BDT"]) {
        if ( (reader -> EvaluateMVA("BDT method")) >     0.10   )    AS_cut[ii][1]++; // -0.35 for t_br and -0.40 for t_bi (test both for -0.1)
      }

//-------------------------------------------------------------------
    } //event loop


// ----------------- Get elapsed time
      sw.Stop();

//===================================================================================================================

std::cout << "================================================================" << std::endl;   

// ----------------- cout efficiencies :

     if(ii==0)  
        {
        std::cout << "----  total                   = " <<  AS_cut[ii][0] << std::endl;
        std::cout << "----  After_CUT               = " <<  AS_cut[ii][1] << std::endl;
        std::cout << "----  MG cross-section [pb]   = " <<  sigma_MG[ii] << std::endl;
        std::cout << "---- tbi efficiency pre    = " <<  eff_pre [ii] << std::endl;
        std::cout << "---- tbi efficiency app    = " <<  AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- tbi efficiency tot >>>>>>>>>>>>>>>>>>  = " <<  eff_pre[ii]*AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- tbi final cross-section [pb] >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*sigma_MG[ii]/ AS_cut[ii][0] << std::endl;
std::cout << "================================================================" << std::endl;   
         }
     if(ii==1)
        {
        std::cout << "----  total                   = " <<  AS_cut[ii][0] << std::endl;
        std::cout << "----  After_CUT               = " <<  AS_cut[ii][1] << std::endl;
        std::cout << "----  MG cross-section [pb]   = " <<  sigma_MG[ii] << std::endl;
        std::cout << "---- ep2bbjvbkg1 efficiency pre    = " <<  eff_pre [ii] << std::endl;
        std::cout << "---- ep2bbjvbkg1 efficiency app    = " <<  AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2bbjvbkg1 efficiency tot >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2bbjvbkg1 final cross-section [pb] >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*sigma_MG[ii]/ AS_cut[ii][0] << std::endl;
std::cout << "================================================================" << std::endl;   
         }
     if(ii==2)
        {
        std::cout << "----  total                   = " <<  AS_cut[ii][0] << std::endl;
        std::cout << "----  After_CUT               = " <<  AS_cut[ii][1] << std::endl;
        std::cout << "----  MG cross-section [pb]   = " <<  sigma_MG[ii] << std::endl;
        std::cout << "---- ep2ccjvbkg2 efficiency pre    = " <<  eff_pre [ii] << std::endl;
        std::cout << "---- ep2ccjvbkg2 efficiency app    = " <<  AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2ccjvbkg2 efficiency tot >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2ccjvbkg2 final cross-section [pb] >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*sigma_MG[ii]/ AS_cut[ii][0] << std::endl;
std::cout << "================================================================" << std::endl;   
         }
     if(ii==3)
        {
        std::cout << "----  total                   = " <<  AS_cut[ii][0] << std::endl;
        std::cout << "----  After_CUT               = " <<  AS_cut[ii][1] << std::endl;
        std::cout << "----  MG cross-section [pb]   = " <<  sigma_MG[ii] << std::endl;
        std::cout << "---- ep2vjz2bbbkg3 efficiency pre    = " <<  eff_pre [ii] << std::endl;
        std::cout << "---- ep2vjz2bbbkg3 efficiency app    = " <<  AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2vjz2bbbkg3 efficiency tot >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2vjz2bbbkg3 final cross-section [pb] >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*sigma_MG[ii]/ AS_cut[ii][0] << std::endl;
std::cout << "================================================================" << std::endl;   
         }
     if(ii==4)
        {
        std::cout << "----  total                   = " <<  AS_cut[ii][0] << std::endl;
        std::cout << "----  After_CUT               = " <<  AS_cut[ii][1] << std::endl;
        std::cout << "----  MG cross-section [pb]   = " <<  sigma_MG[ii] << std::endl;
        std::cout << "---- ep2vjz2ccbkg4 efficiency pre    = " <<  eff_pre [ii] << std::endl;
        std::cout << "---- ep2vjz2ccbkg4 efficiency app    = " <<  AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2vjz2ccbkg4 efficiency tot >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2vjz2ccbkg4 final cross-section [pb] >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*sigma_MG[ii]/ AS_cut[ii][0] << std::endl;
std::cout << "================================================================" << std::endl;   
         }
     if(ii==5)
        {
        std::cout << "----  total                   = " <<  AS_cut[ii][0] << std::endl;
        std::cout << "----  After_CUT               = " <<  AS_cut[ii][1] << std::endl;
        std::cout << "----  MG cross-section [pb]   = " <<  sigma_MG[ii] << std::endl;
        std::cout << "---- ep2vjz2jjbkg5 efficiency pre    = " <<  eff_pre [ii] << std::endl;
        std::cout << "---- ep2vjz2jjbkg5 efficiency app    = " <<  AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2vjz2jjbkg5 efficiency tot >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2vjz2jjbkg5 final cross-section [pb] >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*sigma_MG[ii]/ AS_cut[ii][0] << std::endl;
std::cout << "================================================================" << std::endl;   
         }
     if(ii==6)
        {
        std::cout << "----  total                   = " <<  AS_cut[ii][0] << std::endl;
        std::cout << "----  After_CUT               = " <<  AS_cut[ii][1] << std::endl;
        std::cout << "----  MG cross-section [pb]   = " <<  sigma_MG[ii] << std::endl;
        std::cout << "---- ep2jt2w2jjbkg6 efficiency pre    = " <<  eff_pre [ii] << std::endl;
        std::cout << "---- ep2jt2w2jjbkg6 efficiency app    = " <<  AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2jt2w2jjbkg6 efficiency tot >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2jt2w2jjbkg6 final cross-section [pb] >>>>>>>>>>>>>>>>>>     = " <<  eff_pre[ii]*AS_cut[ii][1]*sigma_MG[ii]/ AS_cut[ii][0] << std::endl;
std::cout << "================================================================" << std::endl;   
         }
     if(ii==7)
        {
        std::cout << "----  total                   = " <<  AS_cut[ii][0] << std::endl;
        std::cout << "----  After_CUT               = " <<  AS_cut[ii][1] << std::endl;
        std::cout << "----  MG cross-section [pb]   = " <<  sigma_MG[ii] << std::endl;
        std::cout << "---- ep2vjhSMbkg7 efficiency pre    = " <<  eff_pre [ii] << std::endl;
        std::cout << "---- ep2vjhSMbkg7 efficiency app     = " <<  AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2vjhSMbkg7 efficiency tot >>>>>>>>>>>>>>>>>>     = " <<  eff_pre[ii]*AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
        std::cout << "---- ep2vjhSMbkg7 final cross-section [pb] >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*sigma_MG[ii]/ AS_cut[ii][0] << std::endl;
std::cout << "================================================================" << std::endl; }  
        
            if(ii==8)
               {
               std::cout << "----  total                   = " <<  AS_cut[ii][0] << std::endl;
               std::cout << "----  After_CUT               = " <<  AS_cut[ii][1] << std::endl;
               std::cout << "----  MG cross-section [pb]   = " <<  sigma_MG[ii] << std::endl;
               std::cout << "---- ep2vjhSMbkg7 efficiency pre    = " <<  eff_pre [ii] << std::endl;
               std::cout << "---- ep2vjhSMbkg7 efficiency app     = " <<  AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
               std::cout << "---- ep2vjhSMbkg7 efficiency tot >>>>>>>>>>>>>>>>>>     = " <<  eff_pre[ii]*AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
               std::cout << "---- ep2vjhSMbkg7 final cross-section [pb] >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*sigma_MG[ii]/ AS_cut[ii][0] << std::endl;
       std::cout << "================================================================" << std::endl;
            
        if(ii==9)
        {
                      std::cout << "----  total                   = " <<  AS_cut[ii][0] << std::endl;
                      std::cout << "----  After_CUT               = " <<  AS_cut[ii][1] << std::endl;
                      std::cout << "----  MG cross-section [pb]   = " <<  sigma_MG[ii] << std::endl;
                      std::cout << "---- ep2vjhSMbkg7 efficiency pre    = " <<  eff_pre [ii] << std::endl;
                      std::cout << "---- ep2vjhSMbkg7 efficiency app     = " <<  AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
                      std::cout << "---- ep2vjhSMbkg7 efficiency tot >>>>>>>>>>>>>>>>>>     = " <<  eff_pre[ii]*AS_cut[ii][1]*1./ AS_cut[ii][0] << std::endl;
                      std::cout << "---- ep2vjhSMbkg7 final cross-section [pb] >>>>>>>>>>>>>>>>>>    = " <<  eff_pre[ii]*AS_cut[ii][1]*sigma_MG[ii]/ AS_cut[ii][0] << std::endl;
              std::cout << "================================================================" << std::endl;
         }

   }  // sample loop
   
std::cout << std::endl;   
std::cout << "*****************************************************************" << std::endl;   
std::cout << "*****************************************************************" << std::endl;   


  std::cout << "---- Final (all) bkg xsec after TMVA Application [pb] = " << ((AS_cut[1][1]*eff_pre[1]*sigma_MG[1]/ AS_cut[1][0])+ // [0] is for signal, bkgs start from [1]
                                                                              (AS_cut[2][1]*eff_pre[2]*sigma_MG[2]/ AS_cut[2][0])+
                                                                              (AS_cut[3][1]*eff_pre[3]*sigma_MG[3]/ AS_cut[3][0])+
                                                                              (AS_cut[4][1]*eff_pre[4]*sigma_MG[4]/ AS_cut[4][0])+
                                                                              (AS_cut[5][1]*eff_pre[5]*sigma_MG[5]/ AS_cut[5][0])+
                                                                              (AS_cut[6][1]*eff_pre[6]*sigma_MG[6]/ AS_cut[6][0])+
                                                                              (AS_cut[7][1]*eff_pre[7]*sigma_MG[7]/ AS_cut[7][0])+
                                                                              (AS_cut[8][1]*eff_pre[8]*sigma_MG[8]/ AS_cut[8][0])+
                                                                              (AS_cut[9][1]*eff_pre[9]*sigma_MG[9]/ AS_cut[9][0]) )
 << endl << std::endl;
 std::cout << "*****************************************************************" << std::endl;   


 //=============================================================================================================================

   // Write histograms

   TFile *target  = new TFile( "FCCeeHlnuqq.root","RECREATE" );

   if (Use["BDTG"])
    {    
       histBdtG[0]->Write();
       histBdtG[1]->Write();
       histBdtG[2]->Write();
       histBdtG[3]->Write();
       histBdtG[4]->Write();
       histBdtG[5]->Write();
       histBdtG[6]->Write();
       histBdtG[7]->Write();
       histBdtG[8]->Write();
       histBdtG[9]->Write();
    }

   target->Close();

   std::cout << "--- Created root file: \"FCCeeHlnuqq.root\" containing the MVA output histograms" << std::endl;

   delete reader;

   std::cout << "==> TMVAClassificationApplication is done!"  << std::endl;
}

 int main( int argc, char** argv )
{
   TString methodList;
   for (int i=1; i<argc; i++) {
      TString regMethod(argv[i]);
      if(regMethod=="-b" || regMethod=="--batch") continue;
      if (!methodList.IsNull()) methodList += TString(",");
      methodList += regMethod;
   }
   ElectronHlnuqq(methodList);
   return 0;
}
