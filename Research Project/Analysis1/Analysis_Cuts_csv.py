import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import ROOT

def significance(s, b):
    if b == 0:
        return 0
    else:
        return math.sqrt(2 * (s + b) * math.log(1 + s / b) - 2 * s)
    
def significance_easy(s, b):
    if b == 0:
        return 0
    else:
        return s/math.sqrt(s+b)

def calculate_efficiency(S_cut, S_total):
    if S_total == 0:
        return 0
    else:
        return S_cut / S_total

data = pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisdata.csv", delimiter=",")

dataframes = {
    "Zee": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisZee.csv", delimiter=","),
    "Zmumu": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisZmumu.csv", delimiter=","),
    "Ztautau": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisZtautau.csv", delimiter=","),
    "Wplusenu": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisWplusenu.csv", delimiter=","),
    "Wplusmunu": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisWplusmunu.csv", delimiter=","),
    "Wplustaunu": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisWplustaunu.csv", delimiter=","),
    "Wminusenu": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisWminusenu.csv", delimiter=","),
    "Wminusmunu": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisWminusmunu.csv", delimiter=","),
    "Wminustaunu": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisWminustaunu.csv", delimiter=","),
    "ZqqZll": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisZqqZll.csv", delimiter=","),
    "WqqZll": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisWqqZll.csv", delimiter=","),
    "WpqqWmlv": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisWpqqWmlv.csv", delimiter=","),
    "WplvWmqq": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisWplvWmqq.csv", delimiter=","),
    "WlvZqq": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisWlvZqq.csv", delimiter=","),
    "llll": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisllll.csv", delimiter=","),
    "lllv": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysislllv.csv", delimiter=","),
    "llvv": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisllvv.csv", delimiter=","),
    "lvvv": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysislvvv.csv", delimiter=","),
    "ttbar_lep": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisttbar_lep.csv", delimiter=","),
    "single_top_tchan": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysissingle_top_tchan.csv", delimiter=","),
    "single_antitop_tchan": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysissingle_antitop_tchan.csv", delimiter=","),
    "single_top_schan": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysissingle_top_schan.csv", delimiter=","),
    "single_antitop_schan": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysissingle_antitop_schan.csv", delimiter=","),
    "single_top_wtchan": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysissingle_top_wtchan.csv", delimiter=","),
    "single_antitop_wtchan": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysissingle_antitop_wtchan.csv", delimiter=","),
    "ggH125_ZZ4lep": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisggH125_ZZ4lep.csv", delimiter=","),
    "ZH125_ZZ4lep": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisZH125_ZZ4lep.csv", delimiter=","),
    "WH125_ZZ4lep": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisWH125_ZZ4lep.csv", delimiter=","),
    "VBFH125_ZZ4lep": pd.read_csv("/home/goncaloribeiro/HZZAnalysis/CSVFiles/Output_HZZAnalysisVBFH125_ZZ4lep.csv", delimiter=",")
}

features = dataframes["ttbar_lep"].columns
plot_features = ['mLL1', 'mLL2', 'fourlepsys_pt', 'fourlepsys_y', 'mass_ext_four_lep',
       'mass_four_lep', 'n_jets']
for name, df in dataframes.items():
    if "H125" in name:
        df['label'] = 1
    else:
        df['label'] = 0

dataframes['llll']['weight'] *= 1.3
dataframes['ZqqZll']['weight'] *= 1.3
dataframes['llvv']['weight'] *= 1.3
lumi = 10064  # pb-1
info = {
    "Zee": (1950.5295, 150277594200),
    "Zmumu": (1950.6321, 147334691090),
    "Ztautau": (1950.6321, 56171652547.3),
    "Wplusenu": (11500.4632, 473389396815),
    "Wplusmunu": (11500.4632, 446507925520),
    "Wplustaunu": (11500.4632, 670928468875),
    "Wminusenu": (8579.63498, 247538642447),
    "Wminusmunu": (8579.63498, 264338188182),
    "Wminustaunu": (8579.63498, 165195850954),
    "ZqqZll": (2.20355112, 3439266.11559),
    "WqqZll": (3.4328, 241438.72705),
    "WpqqWmlv": (24.708, 998250.783475),
    "WplvWmqq": (24.724, 1069526.41899),
    "WlvZqq": (11.42, 1111991.15979),
    "llll": (1.2578, 7538705.8077),
    "lllv": (4.6049, 5441475.00407),
    "llvv": (12.466, 5039259.9696),
    "lvvv": (3.2286, 1727991.07441),
    "ttbar_lep": (452.693559, 49386600),
    "single_top_tchan": (44.152, 4986200),
    "single_antitop_tchan": (26.276, 4989800),
    "single_top_schan": (2.06121, 997800),
    "single_antitop_schan": (1.288662, 995400),
    "single_top_wtchan": (35.845486, 4865800),
    "single_antitop_wtchan": (35.824406, 4945600),
    "ggH125_ZZ4lep": (0.0060239, 27881776.6536),
    "ZH125_ZZ4lep": (0.0000021424784, 150000),
    "WH125_ZZ4lep": (0.0003769, 149400),
    "VBFH125_ZZ4lep": (0.0004633012, 3680490.83243)
}

for name, df in dataframes.items():
    cross_section, total_events = info[name]
    df['scale_weight'] = df['weight'] * lumi * cross_section / total_events

data_MC = pd.concat(dataframes.values(), ignore_index=True)

background_data = data_MC[data_MC['label'] == 0]
signal_data = data_MC[data_MC['label'] == 1]

variables = ["mLL1", 
             "mLL2", 
             "fourlepsys_pt", 
             "fourlepsys_y", 
             "mass_ext_four_lep", 
             "mass_four_lep", 
             "n_jets"]


histogram_ranges = [("mLL1", 50., 106.), 
                    ("mLL2", 12., 140.), 
                    ("fourlepsys_pt", 0., 200.), 
                    ("fourlepsys_y", -3., 3.), 
                    ("mass_ext_four_lep", 80., 250.),
                    ("mass_four_lep", 80., 170.),
                    ("n_jets", -0.5, 3.5)]

histogram_bins = [("mLL1", 30), 
                  ("mLL2", 30), 
                  ("fourlepsys_pt", 20), 
                  ("fourlepsys_y", 20), 
                  ("mass_ext_four_lep", 30),
                  ("mass_four_lep", 24),
                  ("n_jets", 4)]

c = ROOT.TCanvas()

sighist = {}
bkghist = {}
datahist = {}

for var, (var_range, var_min, var_max), (var_bins, bins) in zip(variables, histogram_ranges, histogram_bins):
    print(var)
    sighist[var] = ROOT.TH1F(f"signal_{var}", f"Signal {var}", bins, var_min, var_max)
    bkghist[var] = ROOT.TH1F(f"background_{var}", f"Background {var}", bins, var_min, var_max)
    datahist[var] = ROOT.TH1F(f"data_{var}", f"Data {var}", bins, var_min, var_max)

for var in variables:
    for value, weight in zip(background_data[var], background_data['scale_weight']):
        bkghist[var].Fill(value, weight)
    for value, weight in zip(signal_data[var], signal_data['scale_weight']):
        sighist[var].Fill(value, weight)
    for value in data[var]:
        datahist[var].Fill(value)

for var in variables:
    c.cd()
    
    datahist[var].SetLineColor(ROOT.kBlack)
    datahist[var].SetFillStyle(3004)
    datahist[var].SetStats(0)
    datahist[var].SetTitle(f"{var}")
    datahist[var].Draw("E1 SAME")

    bkghist[var].SetLineColor(ROOT.kBlue)
    bkghist[var].SetFillColor(ROOT.kBlue)
    bkghist[var].SetFillStyle(3004)
    bkghist[var].SetStats(0)
    bkghist[var].Draw("HISTSAME")
    
    sighist[var].SetLineColor(ROOT.kRed)
    sighist[var].SetFillColor(ROOT.kRed)
    sighist[var].SetFillStyle(3005)
    sighist[var].SetStats(0)
    sighist[var].Draw("HISTSAME")
    
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.AddEntry(bkghist[var], "Background", "f")
    legend.AddEntry(sighist[var], "Signal", "f")
    legend.AddEntry(datahist[var], "Data", "l")
    legend.Draw()
    
    c.SaveAs(f"Results/{var}.png")
    c.Clear()

Significance_Right_Cut = {}
Significance_Left_Cut = {}

for var in variables:
    sighist_clone = sighist[var].Clone()
    bkghist_clone = bkghist[var].Clone()

    # Right cut significance
    significance_right = sighist_clone.Clone(f"{var}_right")
    significance_right.Reset()
    S_right = 0
    B_right = 0
    S_left = 0
    B_left = 0

    for j in range(1, sighist_clone.GetNbinsX() + 1):
        S_right = S_right + sighist_clone.GetBinContent(j)
        B_right = B_right + bkghist_clone.GetBinContent(j)
        significance_right.SetBinContent(j, significance(S_right, B_right))

    Significance_Right_Cut[var] = significance_right

    # Left cut significance
    significance_left = sighist_clone.Clone(f"{var}_left")
    significance_left.Reset()
    
    for j in range(1, sighist_clone.GetNbinsX() + 1):
        S_left = S_left + sighist_clone.GetBinContent(sighist_clone.GetNbinsX() + 1 - j)
        B_left = B_left + bkghist_clone.GetBinContent(sighist_clone.GetNbinsX() + 1 - j)
        significance_left.SetBinContent(sighist_clone.GetNbinsX() + 1 - j, significance(S_left, B_left))

    Significance_Left_Cut[var] = significance_left

for var in variables:
    Significance_Right_Cut[var].SetStats(0)
    Significance_Right_Cut[var].GetXaxis().SetTitle(f"{var} [GeV]")
    Significance_Right_Cut[var].GetYaxis().SetTitle(f"Significance 0 -> i")
    Significance_Right_Cut[var].SetTitle(f"Significance for right cut of {var}")
    Significance_Right_Cut[var].Draw()
    c.Print(f"Results/significance_right_{var}.png")
    c.Clear()

for var in variables:
    Significance_Left_Cut[var].SetStats(0)
    Significance_Left_Cut[var].GetXaxis().SetTitle(f"{var} [GeV]")
    Significance_Left_Cut[var].GetYaxis().SetTitle(f"Significance i -> 0")
    Significance_Left_Cut[var].SetTitle(f"Significance for left cut of {var}")
    Significance_Left_Cut[var].Draw()
    c.Print(f"Results/significance_left_{var}.png")
    c.Clear()

left_cuts = {}
right_cuts = {}

for var in variables:
    left_cut = Significance_Left_Cut[var].GetXaxis().GetBinCenter(Significance_Left_Cut[var].GetMaximumBin())
    right_cut = Significance_Right_Cut[var].GetXaxis().GetBinCenter(Significance_Right_Cut[var].GetMaximumBin())
    
    left_cuts[var] = left_cut
    right_cuts[var] = right_cut
    print(f"{var}: {left_cuts[var]:.3f} -> {right_cuts[var]:.3f}")

sighist_cut_1variable = {}
bkghist_cut_1variable = {}

for var in variables:
    sighist_cut_1variable[var] = sighist[var].Clone(f"One_Variable_Cut_Signal_{var}")
    bkghist_cut_1variable[var] = bkghist[var].Clone(f"One_Variable_Cut_Background_{var}")

    for k in range(1,sighist[var].GetNbinsX() + 1):
        bin_center = sighist_cut_1variable[var].GetBinCenter(k)
        if bin_center < left_cuts[var] or bin_center > right_cuts[var]:
            sighist_cut_1variable[var].SetBinContent(k, 0)
            bkghist_cut_1variable[var].SetBinContent(k,0)
    
    sighist_cut_1variable[var].Draw("hist")
    bkghist_cut_1variable[var].Draw("histsame")
    c.Print(f"Results/Cut_{var}.png")
    signal_entries_before_cut = sighist[var].Integral()
    signal_entries_after_cut = sighist_cut_1variable[var].Integral()
    background_entries_before_cut = bkghist[var].Integral()
    background_entries_after_cut = bkghist_cut_1variable[var].Integral()
    significance_before_cuts = significance(signal_entries_before_cut,background_entries_before_cut)
    significance_best_variable_cut = significance(signal_entries_after_cut,background_entries_after_cut)
    signal_effiency_after_cut = calculate_efficiency(signal_entries_after_cut,signal_entries_before_cut)
    print("\n\n")
    print(f"Variable under analysis: {var}")
    print("\n\n")
    print(f"The number of signal events before cut is: {signal_entries_before_cut:.3f}.")
    print(f"The number of background events before cut is: {background_entries_before_cut:.3f}.")
    print(f"The number of signal events after cut is: {signal_entries_after_cut:.3f}.")
    print(f"The number of background events after cut is: {background_entries_after_cut:.3f}.")
    print(f"The signal effiency after cut is: {signal_effiency_after_cut:.3f}.")
    print(f"Before the cut, the significance is {significance_before_cuts:.3f}. After applying the best cut, it is: {significance_best_variable_cut:.3f}.")

data_MC_corr = data_MC.drop(columns=['scale_weight', 'weight', 'label'])
corrmat = data_MC_corr.corr()
heatmap1 = plt.pcolor(corrmat) # get heatmap
plt.colorbar(heatmap1) # plot colorbar
plt.title("Correlations between variables") # set title
x_variables = corrmat.columns.values # get variables from data columns
plt.xticks(np.arange(len(x_variables))+0.5, x_variables, rotation=60) # x-tick for each label
plt.yticks(np.arange(len(x_variables))+0.5, x_variables) # y-tick for each label

filtered_data_MC = data_MC.query(
    'mLL1 >= @left_cuts["mLL1"] and mLL1 <= @right_cuts["mLL1"] and '
    'mLL2 >= @left_cuts["mLL2"] and mLL2 <= @right_cuts["mLL2"] and '
    'fourlepsys_pt >= @left_cuts["fourlepsys_pt"] and fourlepsys_pt <= @right_cuts["fourlepsys_pt"] and '
    'mass_four_lep >= @left_cuts["mass_four_lep"] and mass_four_lep <= @right_cuts["mass_four_lep"]'
)


signal_data = filtered_data_MC[filtered_data_MC['label'] == 1]
background_data = filtered_data_MC[filtered_data_MC['label'] == 0]

# Create ROOT histograms for signal and background
hist_signal = ROOT.TH1F("mass_four_lep_signal", "Mass Four Leptons (Signal)", 24,80,170)
hist_background = ROOT.TH1F("mass_four_lep_background", "Mass Four Leptons (Background)", 24,80,170)

# Fill the histograms with the filtered data, using the 'scale_weight' as the weight
for value, weight in zip(signal_data['mass_four_lep'], signal_data['scale_weight']):
    hist_signal.Fill(value, weight)

for value, weight in zip(background_data['mass_four_lep'], background_data['scale_weight']):
    hist_background.Fill(value, weight)

hist_signal.Draw("hist")
hist_background.Draw("histsame")
print(hist_signal.Integral())
print(hist_background.Integral())
ef = calculate_efficiency(hist_signal.Integral(),sighist['mass_four_lep'].Integral())
significance_all = significance(hist_signal.Integral(),hist_background.Integral())
print(ef)
print(significance_all)
c.Print("mass_four_lep_hist_all_cuts.png")