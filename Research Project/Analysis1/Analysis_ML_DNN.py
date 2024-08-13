import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pyhf
import scipy
import ROOT
import seaborn as sns

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tf.random.set_seed(2)
import random

random.seed(42)

data = pd.read_csv("CSVFiles/Output_HZZAnalysisdata.csv", delimiter=",")

dataframes = {
    "Zee": pd.read_csv("CSVFiles/Output_HZZAnalysisZee.csv", delimiter=","),
    "Zmumu": pd.read_csv("CSVFiles/Output_HZZAnalysisZmumu.csv", delimiter=","),
    "Ztautau": pd.read_csv("CSVFiles/Output_HZZAnalysisZtautau.csv", delimiter=","),
    "Wplusenu": pd.read_csv("CSVFiles/Output_HZZAnalysisWplusenu.csv", delimiter=","),
    "Wplusmunu": pd.read_csv("CSVFiles/Output_HZZAnalysisWplusmunu.csv", delimiter=","),
    "Wplustaunu": pd.read_csv("CSVFiles/Output_HZZAnalysisWplustaunu.csv", delimiter=","),
    "Wminusenu": pd.read_csv("CSVFiles/Output_HZZAnalysisWminusenu.csv", delimiter=","),
    "Wminusmunu": pd.read_csv("CSVFiles/Output_HZZAnalysisWminusmunu.csv", delimiter=","),
    "Wminustaunu": pd.read_csv("CSVFiles/Output_HZZAnalysisWminustaunu.csv", delimiter=","),
    "ZqqZll": pd.read_csv("CSVFiles/Output_HZZAnalysisZqqZll.csv", delimiter=","),
    "WqqZll": pd.read_csv("CSVFiles/Output_HZZAnalysisWqqZll.csv", delimiter=","),
    "WpqqWmlv": pd.read_csv("CSVFiles/Output_HZZAnalysisWpqqWmlv.csv", delimiter=","),
    "WplvWmqq": pd.read_csv("CSVFiles/Output_HZZAnalysisWplvWmqq.csv", delimiter=","),
    "WlvZqq": pd.read_csv("CSVFiles/Output_HZZAnalysisWlvZqq.csv", delimiter=","),
    "llll": pd.read_csv("CSVFiles/Output_HZZAnalysisllll.csv", delimiter=","),
    "lllv": pd.read_csv("CSVFiles/Output_HZZAnalysislllv.csv", delimiter=","),
    "llvv": pd.read_csv("CSVFiles/Output_HZZAnalysisllvv.csv", delimiter=","),
    "lvvv": pd.read_csv("CSVFiles/Output_HZZAnalysislvvv.csv", delimiter=","),
    "ttbar_lep": pd.read_csv("CSVFiles/Output_HZZAnalysisttbar_lep.csv", delimiter=","),
    "single_top_tchan": pd.read_csv("CSVFiles/Output_HZZAnalysissingle_top_tchan.csv", delimiter=","),
    "single_antitop_tchan": pd.read_csv("CSVFiles/Output_HZZAnalysissingle_antitop_tchan.csv", delimiter=","),
    "single_top_schan": pd.read_csv("CSVFiles/Output_HZZAnalysissingle_top_schan.csv", delimiter=","),
    "single_antitop_schan": pd.read_csv("CSVFiles/Output_HZZAnalysissingle_antitop_schan.csv", delimiter=","),
    "single_top_wtchan": pd.read_csv("CSVFiles/Output_HZZAnalysissingle_top_wtchan.csv", delimiter=","),
    "single_antitop_wtchan": pd.read_csv("CSVFiles/Output_HZZAnalysissingle_antitop_wtchan.csv", delimiter=","),
    "ggH125_ZZ4lep": pd.read_csv("CSVFiles/Output_HZZAnalysisggH125_ZZ4lep.csv", delimiter=","),
    "ZH125_ZZ4lep": pd.read_csv("CSVFiles/Output_HZZAnalysisZH125_ZZ4lep.csv", delimiter=","),
    "WH125_ZZ4lep": pd.read_csv("CSVFiles/Output_HZZAnalysisWH125_ZZ4lep.csv", delimiter=","),
    "VBFH125_ZZ4lep": pd.read_csv("CSVFiles/Output_HZZAnalysisVBFH125_ZZ4lep.csv", delimiter=",")
}

features = dataframes["ttbar_lep"].columns
print(features)
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

data_MC_corr = data_MC.drop(columns=['scale_weight', 'weight', 'label'])
corrmat = data_MC_corr.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corrmat, annot=True, cmap='coolwarm', center=0)
plt.title("Correlations between variables")
plt.xticks(np.arange(len(corrmat.columns)) + 0.5, corrmat.columns, rotation=60, ha='right')
plt.yticks(np.arange(len(corrmat.columns)) + 0.5, corrmat.columns, rotation=0)
plt.tight_layout()
plt.savefig('Results_ML/Correlation_map.png')

data_MC['train_weight'] = 1

data_MC.loc[data_MC.query('label==1').index,'train_weight'] = data_MC.loc[data_MC.query('label==1').index,'scale_weight'] \
                                                      / data_MC.loc[data_MC.query('label==1').index,'scale_weight'].sum()
data_MC.loc[data_MC.query('label==0').index,'train_weight'] = data_MC.loc[data_MC.query('label==0').index,'scale_weight'] \
                                                      / data_MC.loc[data_MC.query('label==0').index,'scale_weight'].sum()

background_data = data_MC[data_MC['label'] == 0]
signal_data = data_MC[data_MC['label'] == 1]

sum_w_sig = data_MC.query('label==0')['train_weight'].sum()
sum_w_bkg = data_MC.query('label==1')['train_weight'].sum()


print(f'Sum of weights for training Signal {sum_w_sig:.3} and Background {sum_w_bkg:.3}')
train_features = ['mLL1', 'mLL2', 'fourlepsys_pt', 'fourlepsys_y','mass_four_lep', 'n_jets']

print(f'Data sample x {len(data_MC)} events\n')
x_train,x_val,y_train,y_val,w_train,w_val = train_test_split(data_MC[train_features].values,data_MC['label'].values,
                                                             data_MC['train_weight'].values, train_size=1/3,random_state=9)
print(f'1st split Train sample x {len(x_train)} events (y {len(y_train)} events)')
print(f'1st split Val   sample x {len(x_val)} events (y {len(y_val)} events)\n')

x_val,x_test,y_val,y_test,w_val,w_test = train_test_split(x_val,y_val,w_val,test_size=1/2,random_state=9)
print(f'2nd split Train sample x {len(x_train)} events (y {len(y_train)} events)')
print(f'2nd split Val   sample x {len(x_val)} events (y {len(y_val)} events)')
print(f'2nd split Test  sample x {len(x_test)} events (y {len(y_test)} events)')

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)


model = Sequential()

# The Input and the Hidden Layers as in the exercise
model.add(Dense(80, input_dim = x_train.shape[1], activation='relu'))
model.add(Dense(60, activation='relu'))


# The Output Layer
model.add(Dense(1, activation='sigmoid'))

# Compile the network
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=8e-4), weighted_metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, sample_weight=w_train, validation_data=(x_val,y_val,w_val),
                       epochs=100, batch_size=1024, callbacks=EarlyStopping(patience=4))

y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)

bins = plt.hist(y_train_pred, bins=100, density=True, histtype='step', label='train')
plt.hist(y_val_pred, bins=bins[1], density=True, histtype='step', label='validation')

plt.xlabel('DNN output')
plt.ylabel('Events')
plt.legend(loc='upper right')
plt.savefig('Results_ML/train_validation.png')
plt.clf()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('Results_ML/loss_function.png')
plt.clf()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.savefig('Results_ML/accuracy.png')
plt.clf()

bins = plt.hist(y_train_pred[np.where(y_train == 1)], bins=100, density=True, histtype='step', label='signal train')
plt.hist(y_val_pred[np.where(y_val == 1)], bins=bins[1], density=True, histtype='step', label='signal val')
plt.hist(y_train_pred[np.where(y_train == 0)], bins=100, density=True, histtype='step', label='bkg train')
plt.hist(y_val_pred[np.where(y_val == 0)], bins=bins[1], density=True, histtype='step', label='bkg val')

plt.xlabel('DNN output')
plt.ylabel('Events')
plt.legend(loc='upper right')
plt.savefig('Results_ML/signal_bkg_train_val.png')
plt.clf()

fpr, tpr, _ = roc_curve(y_val, y_val_pred)
auc = roc_auc_score(y_val, y_val_pred)
plt.plot(fpr,tpr,label=f'AUC {auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('Results_ML/ROC_curve.png')
print(f'The AUC is {auc:.4f}')
plt.clf()

test = pd.DataFrame(x_test, columns=train_features)
x_test = scaler.transform(x_test)
y_test_pred = model.predict(x_test)
test['y_test_pred'] = y_test_pred
test['w_test'] = w_test
test['y_test'] = y_test

scale_weight_sum_signal = data_MC.loc[data_MC['label'] == 1, 'scale_weight'].sum() * 3
scale_weight_sum_background = data_MC.loc[data_MC['label'] == 0, 'scale_weight'].sum() * 3

test.loc[test['y_test'] == 1, 'w_test'] *= scale_weight_sum_signal
test.loc[test['y_test'] == 0, 'w_test'] *= scale_weight_sum_background

x_data = scaler.transform(data[train_features].values)
data['y_pred'] = model.predict(x_data)

signal_data = test.query('y_test == 1')['y_test_pred']
background_data = test.query('y_test == 0')['y_test_pred']
signal_weights = test.query('y_test == 1')['w_test']
background_weights = test.query('y_test == 0')['w_test']

plt.hist([signal_data, background_data], bins=100, stacked=True, histtype='stepfilled', color=['red', 'orange'], 
         label=['Signal (test)', 'Background (test)'], weights=[signal_weights, background_weights])
plt.hist(data['y_pred'], bins=100, histtype='step', color='blue', label='Measured Data', weights= data['weight'])
plt.xlabel('DNN output')
plt.ylabel('Events')
plt.legend()
plt.savefig('Results_ML/DNN_output_test_data.png')
plt.yscale('log')
plt.savefig('Results_ML/DNN_output_test_data_log_scale.png')
plt.clf()

S = np.histogram(test.query('y_test == 1')['y_test_pred'], bins = 100, weights= test.query('y_test == 1')['w_test'])
B = np.histogram(test.query('y_test == 0')['y_test_pred'], bins = 100, weights= test.query('y_test == 0')['w_test'])
N = np.histogram(data['y_pred'], bins=100, weights= data['weight'])

model_spec = {'channels': [{'name': 'singlechannel',
              'samples': [
              {'name': 'signal','data': S[0].tolist(),
               'modifiers': [{'data': None, 'name': 'mu', 'type': 'normfactor'}]},
              {'name': 'bkg1','data': B[0].tolist(),
               'modifiers': [{'data': None, 'name': 'bkg_scale', 'type': 'normfactor'}]},
              ]
              }],
              "observations": [{ "name": "singlechannel", "data": N[0].tolist() }],
              "measurements": [{ "name": "Measurement", "config": {"poi": "mu", "parameters": []}}],
              "version": "1.0.0",
}

workspace = pyhf.Workspace(model_spec)
model = workspace.model()

data = N[0].tolist() + model.config.auxdata

test_stat = "q0"
test_poi = 0.

def z_value(CL_b):
  return abs(scipy.stats.norm.isf(CL_b))

CLb_obs, _, CLb_exp = pyhf.infer.hypotest(test_poi, data, model, test_stat=test_stat, return_tail_probs=True, return_expected=True)
print(f"Expected (CL_b) p-value: {CLb_exp:.3e}, significance: {z_value(CLb_exp):.3f} sigma")
print(f"Observed (CL_b) p-value: {CLb_obs:.3e}, significance: {z_value(CLb_obs):.3f} sigma")

np.random.seed(42)
best_pars = pyhf.infer.mle.fit(data=data, pdf=model)
print("\nBest fit parameters:")
for i,p in enumerate(best_pars):
  print(f'\t{model.config.par_order[i]}: {best_pars[i]:.2e}')

test_stat = "qtilde"
test_poi = 1.

poi_values = np.linspace(0.1, 5, 50)
obs_limit, exp_limits = pyhf.infer.intervals.upperlimit(data, model, poi_values, level=0.05)
print(f"\nObserved #mu upper limit (obs): {obs_limit:.3f}, Expected #mu upper limit {exp_limits[2]:.3f}")

model_spec_no_bkg_norm = {'channels': [{'name': 'singlechannel',
              'samples': [
              {'name': 'signal','data': S[0].tolist(),
               'modifiers': [{'data': None, 'name': 'mu', 'type': 'normfactor'}]},
              {'name': 'bkg1','data': B[0].tolist(),
               'modifiers': []},
              ]
              }],
              "observations": [{ "name": "singlechannel", "data": N[0].tolist() }],
              "measurements": [{ "name": "Measurement", "config": {"poi": "mu", "parameters": []}}],
              "version": "1.0.0",
}

workspace = pyhf.Workspace(model_spec_no_bkg_norm)
model = workspace.model()

data = N[0].tolist() + model.config.auxdata

test_stat = "q0"
test_poi = 0.

def z_value(CL_b):
  return abs(scipy.stats.norm.isf(CL_b))

CLb_obs, _, CLb_exp = pyhf.infer.hypotest(test_poi, data, model, test_stat=test_stat, return_tail_probs=True, return_expected=True)
print(f"Expected (CL_b) p-value: {CLb_exp:.3e}, significance: {z_value(CLb_exp):.3f} sigma")
print(f"Observed (CL_b) p-value: {CLb_obs:.3e}, significance: {z_value(CLb_obs):.3f} sigma")


best_pars_no_norm_bkg = pyhf.infer.mle.fit(data=data, pdf=model)
print("\nBest fit parameters:")
for i,p in enumerate(best_pars_no_norm_bkg):
  print(f'\t{model.config.par_order[i]}: {best_pars_no_norm_bkg[i]:.2e}')

test_stat = "qtilde"
test_poi = 1.

poi_values = np.linspace(0.1, 5, 50)
obs_limit, exp_limits = pyhf.infer.intervals.upperlimit(data, model, poi_values, level=0.05)
print(f"\nObserved #mu upper limit (obs): {obs_limit:.3f}, Expected #mu upper limit {exp_limits[2]:.3f}")


sigfile = ROOT.TFile("Signal.root","READ")
bkgfile = ROOT.TFile("Background.root","READ")
datafile = ROOT.TFile("data.root","READ")
sighist = {}
bkghist = {}
datahist = {}
c = ROOT.TCanvas()

histogram_name = [
    "hist_mLL1",
    "hist_mLL2",
    "hist_fourlepsys_pt",
    "hist_fourlepsys_y",
    "mass_four_lep",
    "mass_ext_four_lep",
    "hist_n_jets",
]

plot_features = ['mLL1', 'mLL2', 'fourlepsys_pt', 'fourlepsys_y',
       'mass_four_lep', 'mass_ext_four_lep', 'n_jets']

for var,title in zip(histogram_name,plot_features):
    sighist[var] = sigfile.Get(var)
    bkghist[var] = bkgfile.Get(var)
    datahist[var] = datafile.Get(var)
    datahist[var].SetLineColor(ROOT.kBlack)
    sighist[var].SetLineColor(2)
    sighist[var].SetLineWidth(2)
    bkghist[var].SetLineColor(ROOT.kBlue)
    bkghist[var].SetLineWidth(2)
    sighist[var].SetStats(0)
    bkghist[var].SetStats(0)
    datahist[var].SetStats(0)
    hs = ROOT.THStack(f"hs_{title}", var)
    hs.Add(bkghist[var])
    hs.Add(sighist[var])
    legend = ROOT.TLegend(0.75, 0.75, 0.9, 0.9)
    legend.AddEntry(sighist[var], f"HZZ Signal","f")
    legend.AddEntry(bkghist[var], "Background","f")
    legend.AddEntry(datahist[var], "Data","l")
    hs.SetTitle(f"{title}")
    max_y = max(h.GetMaximum() for h in [bkghist[var], sighist[var], datahist[var]])
    hs.SetMinimum(0)
    hs.SetMaximum(max_y * 1.3)  # Add a little padding to the top of the y-axis range
    hs.Draw("hist")
    datahist[var].Draw("e1 same")
    legend.Draw()
    c.Print(f"Results_mu_1/{title}.png")
    c.Clear()

for var,title in zip(histogram_name,plot_features):
    sighist[var] = sigfile.Get(var)
    bkghist[var] = bkgfile.Get(var)
    datahist[var] = datafile.Get(var)
    datahist[var].SetLineColor(ROOT.kBlack)
    sighist[var].SetLineColor(2)
    sighist[var].SetLineWidth(2)
    bkghist[var].SetLineColor(ROOT.kBlue)
    bkghist[var].SetLineWidth(2)
    sighist[var].SetStats(0)
    bkghist[var].SetStats(0)
    datahist[var].SetStats(0)
    sighist[var].Scale(best_pars_no_norm_bkg[0])
    hs = ROOT.THStack(f"hs_{title}", var)
    hs.Add(bkghist[var])
    hs.Add(sighist[var])
    max_y = max(h.GetMaximum() for h in [bkghist[var], sighist[var], datahist[var]])
    hs.SetMinimum(0)
    hs.SetMaximum(max_y * 1.3)  # Add a little padding to the top of the y-axis range
    legend = ROOT.TLegend(0.75, 0.75, 0.9, 0.9)
    legend.AddEntry(sighist[var], f"HZZ Signal mu = {best_pars_no_norm_bkg[0]:.2f}","f")
    legend.AddEntry(bkghist[var], f"Background","f")
    legend.AddEntry(datahist[var], "Data","l")
    hs.SetTitle(f"{title} with mu = {best_pars_no_norm_bkg[0]:.2f}")
    hs.Draw("hist")
    datahist[var].Draw("e1 same")
    legend.Draw()
    c.Print(f"Results_Scaled/{title}_scaled_mu_{best_pars_no_norm_bkg[0]:.2f}.png")
    c.Clear()