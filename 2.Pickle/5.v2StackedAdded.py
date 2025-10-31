import os
import pandas as pd
import numpy as np
import ROOT
import matplotlib.pyplot as plt

# ------------------ INPUT ------------------
input_pkl = "bdt_distributions/bdt_scores_unbinned.pkl"
df = pd.read_pickle(input_pkl)
print(f"✅ Loaded {len(df)} events from {input_pkl}")

# ------------------ OUTPUT DIRECTORY ------------------
output_dir = "binnedoutput"
os.makedirs(output_dir, exist_ok=True)

# ------------------ SAMPLE INFO ------------------
samples_info = {
    "wzp6_ee_Henueqq_ecm125.root": {"cross_section_init": 0.41e-05, "n_gen": 1000000},
    "wzp6_ee_Hqqenue_ecm125.root": {"cross_section_init": 0.41e-05, "n_gen": 1000000},
    "wzp6_ee_Hmunumuqq_ecm125.root": {"cross_section_init": 0.41e-05, "n_gen": 1000000},
    "wzp6_ee_Hqqmunumu_ecm125.root": {"cross_section_init": 0.41e-05, "n_gen": 1000000},
    "wzp6_ee_Htaunutauqq_ecm125.root": {"cross_section_init": 0.41e-05, "n_gen": 1000000},
    "wzp6_ee_Hqqtaunutau_ecm125.root": {"cross_section_init": 0.41e-05, "n_gen": 1000000},
    "wzp6_ee_enueqq_ecm125.root": {"cross_section_init": 2.613e-02, "n_gen": 99600000},
    "wzp6_ee_eeqq_ecm125.root": {"cross_section_init": 3.934, "n_gen": 99800000},
    "wzp6_ee_munumuqq_ecm125.root": {"cross_section_init": 6.711e-03, "n_gen": 99800000},
    "wzp6_ee_mumuqq_ecm125.root": {"cross_section_init": 1.505e-1, "n_gen": 100000000},
    "wzp6_ee_taunutauqq_ecm125.root": {"cross_section_init": 6.761e-03, "n_gen": 99400000},
    "wzp6_ee_tautauqq_ecm125.root": {"cross_section_init": 1.476e-1, "n_gen": 98965252},
    "p8_ee_ZZ_4tau_ecm125.root": {"cross_section_init": 0.003, "n_gen": 100000000},
    "wzp6_ee_Htautau_ecm125.root": {"cross_section_init": 1.011e-4, "n_gen": 10000000},
    "wzp6_ee_Hllnunu_ecm125.root": {"cross_section_init": 3.187e-05, "n_gen": 1200000},
    "wzp6_ee_eenunu_ecm125.root": {"cross_section_init": 6.574e-01, "n_gen": 100000000},
    "wzp6_ee_mumununu_ecm125.root": {"cross_section_init": 2.202e-01, "n_gen": 99400000},
    "wzp6_ee_tautaununu_ecm125.root": {"cross_section_init": 4.265e-02, "n_gen": 99900000},
    "wzp6_ee_l1l2nunu_ecm125.root": {"cross_section_init": 9.845e-03, "n_gen": 99500000},
    "wzp6_ee_tautau_ecm125.root": {"cross_section_init": 25.939, "n_gen": 10000000},
    "wzp6_ee_Hgg_ecm125.root": {"cross_section_init": 7.384e-05, "n_gen": 1200000},
    "wzp6_ee_Hbb_ecm125.root": {"cross_section_init": 1.685e-3, "n_gen": 9900000},
    "wzp6_ee_qq_ecm125.root": {"cross_section_init": 3.631e+02, "n_gen": 498704128},
}

# ------------------ SETTINGS ------------------
n_bins = 10
luminosity = 1e7
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# ------------------ COMPUTE ------------------
results = []
root_output_path = os.path.join(output_dir, "bdt_reco_histos.root")
root_output = ROOT.TFile(root_output_path, "RECREATE")

for process, group in df.groupby("process"):
    info = samples_info.get(process)
    if info is None:
        print(f"⚠️ Skipping {process}: no initial cross section info found.")
        continue

    n_total = group["total_events_in_file"].iloc[0]
    sigma_init = info["cross_section_init"]
    n_gen = info["n_gen"]
    efficiency = n_total / n_gen
    sigma_final = sigma_init * efficiency

    counts, _ = np.histogram(group["bdt_score"], bins=bin_edges)
    fractions = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
    bin_xs = sigma_final * fractions
    reco_events = bin_xs * luminosity

    df_bin = pd.DataFrame({
        "process": process,
        "bin_center": bin_centers,
        "bin_count": counts,
        "bin_fraction": fractions,
        "preselection_efficiency": efficiency,
        "initial_cross_section": sigma_init,
        "final_cross_section": sigma_final,
        "bin_cross_section": bin_xs,
        "luminosity": luminosity,
        "reco_level_events": reco_events
    })
    results.append(df_bin)

    hist_name = process.replace(".root", "")
    hist = ROOT.TH1F(hist_name, f"Reco-level events for {process}", n_bins, 0, 1)
    for i, val in enumerate(reco_events):
        hist.SetBinContent(i + 1, val)
    hist.GetXaxis().SetTitle("BDT Score")
    hist.GetYaxis().SetTitle("Reco-level events")
    hist.Write()

    plt.figure(figsize=(20, 22),dpi=300)
    plt.bar(bin_centers, reco_events, width=0.1, align='center', color='royalblue', alpha=0.7, edgecolor='black')
    plt.yscale("log")
    plt.xlabel("BDT Score")
    plt.ylabel("Reco-level events")
    plt.title(f"{process}\n(L={luminosity:.1e} fb⁻¹)")
    plt.grid(True, which="both", ls="--", lw=0.4, alpha=0.7)
    plt.tight_layout()
    safe_name = hist_name.replace("/", "_")
    plot_path = os.path.join(output_dir, f"{safe_name}_reco_log.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"🖼️ Saved histogram plot for {process} → {plot_path}")

# ------------------ STACKED HISTOGRAM ------------------
stack = ROOT.THStack("bdt_stack", "Stacked Reco-level Events;BDT Score;Reco-level events")
total_hist = ROOT.TH1F("total_reco", "Total Reco-level Events;BDT Score;Reco-level events", n_bins, 0, 1)
colors = [ROOT.kAzure+1, ROOT.kRed+1, ROOT.kGreen+2, ROOT.kOrange+7, ROOT.kMagenta+1,
          ROOT.kViolet-4, ROOT.kCyan+1, ROOT.kYellow+1, ROOT.kPink+7, ROOT.kTeal+3]
color_index = 0

root_output.cd()
for key in root_output.GetListOfKeys():
    obj = key.ReadObj()
    if isinstance(obj, ROOT.TH1) and obj.GetName() not in ["bdt_stack", "total_reco"]:
        obj.SetFillColor(colors[color_index % len(colors)])
        obj.SetLineColor(colors[color_index % len(colors)])
        stack.Add(obj)
        total_hist.Add(obj)
        color_index += 1

stack.Write()
total_hist.Write()
print("✅ Added stacked and total histograms to ROOT file")

c = ROOT.TCanvas("c", "c", 900, 700)
c.SetLogy()
stack.Draw("HIST")
stack.GetXaxis().SetTitle("BDT Score")
stack.GetYaxis().SetTitle("Reco-level events")
c.BuildLegend()
stack_png = os.path.join(output_dir, "stacked_bdt_log.png")
c.SaveAs(stack_png)
c.Close()
print(f"📊 Saved stacked plot → {stack_png}")

# ------------------ SAVE BINNED DATAFRAME ------------------
final_binned_df = pd.concat(results, ignore_index=True)
output_pkl = os.path.join(output_dir, "bdt_binned_with_cross_sections.pkl")
final_binned_df.to_pickle(output_pkl)

root_output.Close()

print(f"\n📦 Saved binned DataFrame → {output_pkl}")
print(f"📊 ROOT histograms saved → {root_output_path}")
print(f"✅ Shape: {final_binned_df.shape}")
print("\nPreview:")
print(final_binned_df.head())

print("\n📈 To view in ROOT:")
print(f"root -l {root_output_path}")
print(">>> bdt_stack->Draw(\"HIST\"); gPad->SetLogy();")
