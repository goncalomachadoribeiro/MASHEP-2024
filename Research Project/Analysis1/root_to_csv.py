import ROOT

def histogram_to_csv(histogram, csv_filename):
    # Open the CSV file
    with open(csv_filename, "w") as csv_file:
        # Write header
        csv_file.write("BinIndex,BinContent\n")

        # Loop over bins in the histogram
        for i in range(1, histogram.GetNbinsX() + 1):
            bin_index = i
            bin_content = histogram.GetBinContent(i)
            csv_file.write(f"{bin_index},{bin_content}\n")

    print("Conversion complete.")

if __name__ == "__main__":
    # Open the ROOT file
    root_file = ROOT.TFile("../data2csv/background.root")

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

    # Convert each histogram to CSV
    for histogram_name in histogram_names:
        histogram = root_file.Get(histogram_name)
        if histogram:
            csv_filename = f"{histogram_name}_bkg.csv"
            histogram_to_csv(histogram, csv_filename)
        else:
            print(f"Warning: Histogram '{histogram_name}' not found in ROOT file.")

    # Close the ROOT file
    root_file.Close()

