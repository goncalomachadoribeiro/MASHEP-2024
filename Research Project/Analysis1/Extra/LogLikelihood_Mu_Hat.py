import scipy
import numpy as np
import math
import pandas as pd
import ROOT

# Read the CSV files
signal_df = pd.read_csv('root_to_csv/Signal_CSV/mass_four_lep_signal.csv')
background_df = pd.read_csv('root_to_csv/Bkg_CSV/mass_four_lep_bkg.csv')
data_df = pd.read_csv('root_to_csv/Data_CSV/mass_four_lep_data.csv')

# Extract the 'BinContent' column into lists
s = signal_df['BinContent'].tolist()
b = background_df['BinContent'].tolist()
n = data_df['BinContent'].tolist()

def likelihood_function(data, mu):
    likelihood_result = 1
    for i in range(len(data)):
        lmbda = s[i] * mu + b[i]
        likelihood_result *= scipy.stats.poisson.pmf(data[i], lmbda)
    return likelihood_result

mu_list = np.arange(0., 5.5, 0.01)
log_likelihood = []

for mu in mu_list:
    log_likelihood.append(np.log(likelihood_function(n, mu)))

# Plotting with ROOT
# Convert lists to ROOT arrays
mu_list_root = np.array(mu_list, dtype=float)
log_likelihood_root = np.array(log_likelihood, dtype=float)

# Create a TGraph
graph = ROOT.TGraph(len(mu_list_root), mu_list_root, log_likelihood_root)

# Create a canvas and draw the graph
canvas = ROOT.TCanvas("canvas", "Log-Likelihood as a function of mu", 800, 600)
graph.SetTitle("Log-Likelihood as a function of #mu;#mu;log-likelihood")
graph.SetLineColor(ROOT.kRed)  # Set line color to magenta
graph.Draw("AL")

# Find the index of maximum likelihood
indexmax = np.argmax(np.array(log_likelihood))
muhat = mu_list[indexmax]

# Draw a vertical line at muhat
line = ROOT.TLine(muhat, graph.GetYaxis().GetXmin(), muhat, graph.GetYaxis().GetXmax())
line.SetLineColor(ROOT.kBlue)  # Set line color to purple
line.SetLineWidth(2)
line.Draw()

# Draw labels and legend
canvas.Update()
canvas.Draw()

# Save the canvas as a PNG file
canvas.SaveAs("log_likelihood.png")

print("$\hat{\mu}$ = ", muhat)



factor = [0.5, 2., 4.5]
std_dev = [1., 2., 3.]

for i in range(len(factor)):
    half_log_likelihood = max(log_likelihood) - factor[i]

    # Find the index where log_likelihood is closest to half_log_likelihood
    index_half_log_likelihood = np.argmin(np.abs(np.array(log_likelihood) - half_log_likelihood))

    # Calculate the uncertainty
    uncertainty = abs(muhat - mu_list[index_half_log_likelihood])

    print("$\\hat{{\\mu}}$ for {:.1f} standard deviation(s): {:.2f} $\\pm$ {:.2f}".format(std_dev[i], muhat, uncertainty))
