import ROOT
import math
import numpy as np
from ROOT.Math import poisson_pdf
import scipy

# Define the significance calculation function
def significance(s, b):
    if b == 0:
        return 0
    else:
        return math.sqrt(2 * (s + b) * math.log(1 + s / b) - 2 * s)

# Function to calculate the selection efficiency
def calc_efficiency(Ntot, Nsel):
    return Nsel / Ntot if Ntot > 0 else 0

# Open the ROOT files
signal_file = ROOT.TFile("../data2csv/Signal.root", "READ")
background_file = ROOT.TFile("../data2csv/Background.root", "READ")
data_file = ROOT.TFile("../data2csv/data.root", "READ")

# List of histogram names
histogram_names = [
    "hist_mLL1",
    "hist_mLL2",
    "hist_fourlepsys_pt",
    "hist_fourlepsys_y",
    "mass_four_lep",
    "mass_ext_four_lep",
    "hist_n_jets",
    "hist_fourleptpt",
    "hist_fourlepteta",
    "hist_fourleptE",
    "hist_fourleptphi",
    "hist_fourleptID"
]

# Create a canvas
c = ROOT.TCanvas()

# Daltonic-friendly colors for histograms
daltonic_colors = [
    ROOT.kBlue,
    ROOT.kRed,
    ROOT.kGreen + 2,
    ROOT.kMagenta,
    ROOT.kOrange + 7
]

# Initialize lists to hold histograms and legends
sighist = [None] * len(histogram_names)
bkghist = [None] * len(histogram_names)
datahist = [None] * len(histogram_names)
legend = []

# Function to draw the histograms
def draw_histogram(i, histogram_name, color):
    # Retrieve histograms from the ROOT files
    sighist[i] = signal_file.Get(histogram_name)
    bkghist[i] = background_file.Get(histogram_name)
    datahist[i] = data_file.Get(histogram_name)
    
    if not sighist[i] or not bkghist[i] or not datahist[i]:
        print(f"Error: Histogram '{histogram_name}' not found in one of the files.")
        return

    # Set histogram styles
    sighist[i].SetLineColor(color)
    sighist[i].SetFillColor(color)
    sighist[i].SetFillStyle(3001)  # Solid fill style

    bkghist[i].SetLineColor(color + 1)
    bkghist[i].SetFillColor(color + 1)
    bkghist[i].SetFillStyle(3002)  # Hatched fill style
    
    datahist[i].SetLineColor(ROOT.kBlack)
    datahist[i].SetMarkerStyle(20)  # Points

    sighist[i].SetStats(0)
    bkghist[i].SetStats(0)
    datahist[i].SetStats(0)

    # Draw histograms
    bkghist[i].Draw("hist")
    sighist[i].Draw("histsame")
    datahist[i].Draw("esame")

    # Create and draw legend
    leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    leg.SetHeader(histogram_name, "C")  # Set header to the name of the histogram
    leg.AddEntry(sighist[i], "Signal")
    leg.AddEntry(bkghist[i], "Background")
    leg.AddEntry(datahist[i], "Data")
    leg.Draw()

    # Save and clear canvas
    c.Print(f"{histogram_name}.png")
    c.Clear()

# Loop through histogram names and draw them
for i in range(len(histogram_names)):
    draw_histogram(i, histogram_names[i], daltonic_colors[i % len(daltonic_colors)])

# -------------------------------------------------------------------------------------- #
# Exercise 3) 
# Choosing the 'chosen' variables
best_variables_index = [0,1,2,3,4,5,6,7,8,9,10,11]

# Create Canvas
c.Divide(6, 2)

# Right Cut Significance
Significance_Right_Cut = []
Significance_Left_Cut = []

for i, var_index in enumerate(best_variables_index):
    # Clone the histograms
    sighist_clone = sighist[var_index].Clone()
    bkghist_clone = bkghist[var_index].Clone()

    # Right cut significance
    significance_right = sighist_clone.Clone(f"{histogram_names[var_index]}_right")
    significance_right.Reset()

    for j in range(1, sighist_clone.GetNbinsX() + 1):
        S_right = sighist_clone.Integral(1, j)
        B_right = bkghist_clone.Integral(1, j)
        significance_right.SetBinContent(j, significance(S_right, B_right))

    Significance_Right_Cut.append(significance_right)

    # Left cut significance
    significance_left = sighist_clone.Clone(f"{histogram_names[var_index]}_left")
    significance_left.Reset()
    
    for j in range(1, sighist_clone.GetNbinsX() + 1):
        S_left = sighist_clone.Integral(j, sighist_clone.GetNbinsX())
        B_left = bkghist_clone.Integral(j, bkghist_clone.GetNbinsX())
        significance_left.SetBinContent(j, significance(S_left, B_left))

    Significance_Left_Cut.append(significance_left)

    # Print initial total significance
    initial_significance_right = Significance_Right_Cut[i].Integral()
    initial_significance_left = Significance_Left_Cut[i].Integral()
    print(f"Initial Total Significance for {histogram_names[var_index]}:")
    print(f"  Right Cut: {initial_significance_right:.3f}")
    print(f"  Left Cut: {initial_significance_left:.3f}")

# Draw Right Cut Significance
for i, sig in enumerate(Significance_Right_Cut):
    c.cd(i + 1)
    sig.SetStats(0)
    sig.GetXaxis().SetTitle(f"{histogram_names[best_variables_index[i]]} [GeV]")
    sig.GetYaxis().SetTitle(f"Significance 0 -> i")
    sig.Draw("hist")

c.Print("significanceRight.pdf")
c.Clear()

# Draw Left Cut Significance
c.Divide(6, 2)
for i, sig in enumerate(Significance_Left_Cut):
    c.cd(i + 1)
    sig.SetStats(0)
    sig.GetXaxis().SetTitle(f"{histogram_names[best_variables_index[i]]} [GeV]")
    sig.GetYaxis().SetTitle(f"Significance i -> N")
    sig.Draw("hist")

c.Print("significanceLeft.pdf")
c.Clear()

# Print results
left_cuts = [0.0] * len(best_variables_index)
right_cuts = [0.0] * len(best_variables_index)

print(f"The best cuts are:")
for i, var_index in enumerate(best_variables_index):
    left_cuts[i] = Significance_Left_Cut[i].GetXaxis().GetBinCenter(Significance_Left_Cut[i].GetMaximumBin())
    right_cuts[i] = Significance_Right_Cut[i].GetXaxis().GetBinCenter(Significance_Right_Cut[i].GetMaximumBin())
    print(f"{histogram_names[var_index]}: {left_cuts[i]:.3f} -> {right_cuts[i]:.3f}")

    # Print final total significance after cuts
    final_significance_left = sig.Integral(sig.GetXaxis().FindBin(left_cuts[i]), sig.GetNbinsX())
    final_significance_right = sig.Integral(1, sig.GetXaxis().FindBin(right_cuts[i]))
    print(f"Final Total Significance for {histogram_names[best_variables_index[i]]}:")
    print(f"  Left Cut: {final_significance_left:.3f}")
    print(f"  Right Cut: {final_significance_right:.3f}")


# Define the variable for mT (transverse invariant mass of the WW pair)
mt_var = 'hist_fourlepteta'

# Assuming mt_var corresponds to the name of the variable you want to use for selection cuts
# Create histograms for signal and background after applying selection conditions
sighistSEL = sighist[4].Clone(f"{mt_var}sigSEL")
bkghistSEL = bkghist[4].Clone(f"{mt_var}bkgSEL")
datahistSEL = datahist[4].Clone(f"{mt_var}bkgSEL")

# Apply selection cuts
for bin in range(1, sighistSEL.GetNbinsX() + 1):
    if sighistSEL.GetXaxis().GetBinCenter(bin) > right_cuts[4] or sighistSEL.GetXaxis().GetBinCenter(bin) < left_cuts[4]:
        sighistSEL.SetBinContent(bin, 0)
    if bkghistSEL.GetXaxis().GetBinCenter(bin) > right_cuts[4] or bkghistSEL.GetXaxis().GetBinCenter(bin) < left_cuts[4]:
        bkghistSEL.SetBinContent(bin, 0)
    if datahistSEL.GetXaxis().GetBinCenter(bin) > right_cuts[4] or datahistSEL.GetXaxis().GetBinCenter(bin) < left_cuts[4]:
        datahistSEL.SetBinContent(bin, 0)

# Pattern colors for histograms
signal_pattern_color = ROOT.kBlue
background_pattern_color = ROOT.kRed

# Aesthetics for signal histogram
sighistSEL.SetLineColor(ROOT.kBlack)
sighistSEL.SetMarkerColor(ROOT.kBlack)
sighistSEL.SetMarkerStyle(20)  # Points
sighistSEL.SetMarkerSize(0.8)

# Aesthetics for background histogram
bkghistSEL.SetLineColor(ROOT.kBlack)
bkghistSEL.SetFillColor(background_pattern_color)
bkghistSEL.SetFillStyle(3353)  # Crosshatch pattern
bkghistSEL.SetLineWidth(2)

# Aesthetics for data histogram
datahistSEL.SetLineColor(ROOT.kBlack)
datahistSEL.SetMarkerStyle(20)  # Points
datahistSEL.SetMarkerSize(0.8)

# Axis labels and titles
bkghistSEL.GetXaxis().SetTitle(f"{mt_var} [GeV]")
bkghistSEL.GetYaxis().SetTitle("Events")

# Legend for the histogram
legendSEL = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
legendSEL.SetHeader(mt_var, "C")
legendSEL.AddEntry(sighistSEL, "Signal")
legendSEL.AddEntry(bkghistSEL, "Background")
legendSEL.AddEntry(datahistSEL, "Data")

# Draw histograms on the same canvas
datahistSEL.Draw("e")
bkghistSEL.Draw("histsame")
sighistSEL.Draw("histsame")
legendSEL.Draw()

# Save the canvas as an image file
c.Print(f"{mt_var}SEL.png")

# Extract the 'BinContent' column into lists
# Extract signal histogram content
s = []
for i in range(1, sighistSEL.GetNbinsX() + 1):
    s.append(sighistSEL.GetBinContent(i))

b = []
for i in range(1, bkghistSEL.GetNbinsX() + 1):
    b.append(bkghistSEL.GetBinContent(i))

n = []
for i in range(1, datahistSEL.GetNbinsX() + 1):
    n.append(datahistSEL.GetBinContent(i))



def log_likelihood_function(data, mu):
    likelihood_result = 0
    for i in range(len(data)):
        if (s[i]>0 and b[i]>0):
            lmbda = s[i] * mu + b[i]
            likelihood_result += np.log(scipy.stats.poisson.pmf(data[i], lmbda))
    return likelihood_result

mu_list = np.arange(0., 5.5, 0.01)
log_likelihood = []

for mu in mu_list:
    log_likelihood.append(log_likelihood_function(n, mu))

# Plotting with ROOT
# Convert lists to ROOT arrays
mu_list_root = np.array(mu_list, dtype=float)
log_likelihood_root = np.array(log_likelihood, dtype=float)

# Create a TGraph
graph = ROOT.TGraph(len(mu_list_root), mu_list_root, log_likelihood_root)

# Create a canvas and draw the graph
canvas = ROOT.TCanvas("canvas", "Log-Likelihood as a function of mu", 800, 600)
graph.SetTitle("Log-Likelihood as a function of #mu;#mu;log-likelihood")
graph.SetLineColor(ROOT.kMagenta)
graph.Draw("AL")

# Show the canvas
canvas.Update()
canvas.Draw()

# Save the canvas as a PNG file
canvas.SaveAs("log_likelihood.png")

indexmax = np.argmax(np.array(log_likelihood))
muhat = mu_list[indexmax]

print("$\hat{\mu}$ = ", muhat)

factor = [0.5]
std_dev = [1.]

for i in range(len(factor)):
    half_log_likelihood = max(log_likelihood) - factor[i]

    # Find the index where log_likelihood is closest to half_log_likelihood
    index_half_log_likelihood = np.argmin(np.abs(np.array(log_likelihood) - half_log_likelihood))

    # Calculate the uncertainty
    uncertainty = abs(muhat - mu_list[index_half_log_likelihood])

    print("$\\hat{{\\mu}}$ for {:.1f} standard deviation(s): {:.2f} $\\pm$ {:.2f}".format(std_dev[i], muhat, uncertainty))

    sum_s = 0
    sum_b = 0

    for i in range(len(s)):
        sum_s += s[i]
        sum_b += b[i]

    expected_significance_1 = sum_s / np.sqrt(sum_s + sum_b)
    expected_significance_2 = np.sqrt(2 * (sum_s + sum_b) * np.log(1 + sum_s / sum_b) - 2 * sum_s)

    print("Expected significance for a counting experiment:", expected_significance_1)
    print("Expected significance 'more accurate': ", expected_significance_2)


    sum_s = 0
    sum_b = 0
    sum_n = 0
    

    for i in range(len(s)):
        sum_s += muhat*s[i]
        sum_b += b[i]
        sum_n += n[i]

    observed_significance_1 = sum_s / np.sqrt(sum_n)
    observed_significance_2 = np.sqrt(2 * (sum_n) * np.log(1 + sum_s / sum_b) - 2 * sum_s)

    print("Observed significance for a counting experiment: ", observed_significance_1)
    print("Observed significance 'more accurate': ", observed_significance_2)


# -------------------- #
cut_values = []
significance_values = []

for bin in range(1, sighist[4].GetNbinsX() + 1):
    cut_value = sighist[4].GetXaxis().GetBinCenter(bin)
    S_cut = sighist[4].Integral(1, bin)
    B_cut = bkghist[4].Integral(1, bin)
    cut_values.append(cut_value)
    significance_values.append(significance(S_cut, B_cut))

best_cut_index = int(np.argmax(significance_values)) + 1  # Ensure valid bin index for Integral method
best_cut = cut_values[best_cut_index - 1]

# Calculate the signal efficiency for the best cut value
total_signal_events = sighist[4].Integral(0, sighist[4].GetNbinsX() + 1)  # Include underflow and overflow bins
selected_signal_events = sighist[4].Integral(1, best_cut_index)
signal_efficiency = selected_signal_events / total_signal_events

print(f"The best signal selection criterion based on {mt_var} is mt < {best_cut:.2f} GeV")
print(f"N TOTAL: {total_signal_events}")
print(f"The corresponding signal efficiency is {signal_efficiency:.2f}")

