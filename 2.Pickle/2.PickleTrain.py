import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ==== CONFIG ====
data_dir = "."  # current working directory
signal_files = [
    os.path.join(data_dir, "wzp6_ee_Hqqmunumu_ecm125.pkl"),
    os.path.join(data_dir, "wzp6_ee_Hqqtaunutau_ecm125.pkl"),
]

features = [
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
    #"Jet_nconst0","Jet_nconst1",
    "displacementdz0","displacementdxy0","displacementdz1","displacementdxy1",
    "scoreG1","scoreG2","scoreU1","scoreU2","scoreS1","scoreS2","scoreC1","scoreC2","scoreB1","scoreB2",
    "scoreT1","scoreT2","scoreD1","scoreD2",
    "scoreSumG","scoreSumU","scoreSumS","scoreSumC","scoreSumB","scoreSumT","scoreSumD",
    "scoreMultiplyG","scoreMultiplyU","scoreMultiplyS","scoreMultiplyC","scoreMultiplyB","scoreMultiplyT","scoreMultiplyD"
]

# ==== HELPER FUNCTIONS ====
def load_pickle_to_df(file_list, label_value):
    dfs = []
    for fn in file_list:
        print(f"  Checking {fn} ...")
        if not os.path.exists(fn):
            print(f"  [ERROR] Missing file: {fn}")
            continue
        try:
            print(f"  Reading {fn}")
            df = pd.read_pickle(fn)
            print(f"    Shape: {df.shape}")
            df["label"] = label_value
            dfs.append(df)
        except Exception as e:
            print(f"  [ERROR] Failed to read {fn}: {e}")
    if len(dfs) == 0:
        raise RuntimeError("No valid DataFrames loaded!")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  --> Combined DataFrame shape: {combined.shape}")
    return combined

# ==== MAIN ====
print("=== Collecting all pickle files ===")
all_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
print(f"Found {len(all_files)} .pkl files")

background_files = [f for f in all_files if f not in signal_files]
print(f"Signal files: {len(signal_files)}, Background files: {len(background_files)}")

# --- Load data ---
print("\n=== Loading SIGNAL ===")
df_signal = load_pickle_to_df(signal_files, label_value=1)

print("\n=== Loading BACKGROUND ===")
df_background = load_pickle_to_df(background_files, label_value=0)

# --- Merge ---
print("\n=== Merging dataframes ===")
df_all = pd.concat([df_signal, df_background], ignore_index=True)
print(f"Total events loaded: {len(df_all):,}")
print(f"Memory usage: {df_all.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# --- Clean columns ---
print("\n=== Checking column data types ===")
bad_cols = [c for c in df_all.columns if not pd.api.types.is_numeric_dtype(df_all[c])]
print(f"Non-numeric columns detected: {bad_cols}")

if bad_cols:
    for c in bad_cols:
        print(f"  Cleaning column: {c}")
        df_all[c] = df_all[c].apply(
            lambda x: np.mean(x) if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0
            else (x if isinstance(x, (int, float, np.number)) else 0)
        )
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce").fillna(0)
    print("  -> All columns converted to numeric")

# --- Check feature coverage ---
missing_features = [f for f in features if f not in df_all.columns]
if missing_features:
    print(f"[WARNING] Missing features: {missing_features}")
else:
    print("All expected features found!")

# --- Training set ---
print("\n=== Splitting data ===")
X = df_all[features].copy()
y = df_all["label"]
print(f"X shape: {X.shape}, y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# --- XGBoost Training ---
print("\n=== Starting XGBoost training ===")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.001,
    "eval_metric": "auc",
    "tree_method": "hist",
    "nthread": 64
}

evals = [(dtrain, "train"), (dtest, "test")]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=30000,
    evals=evals,
    early_stopping_rounds=100,
    verbose_eval=50
)

print("\n=== Training complete! ===")
bst.save_model("off-shell-mu.json")
print("Model saved as off-shell-muon.json")

