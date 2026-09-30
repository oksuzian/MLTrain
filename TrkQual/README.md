# TrkQual

## Introduction
The TrkQual algorithm is trained to classify tracks as either "high quality" or "low quality". At the moment, the model implemented in Offline is an Artificial Neural Network (see [here](https://github.com/Mu2e/Offline/blob/main/TrkDiag/src/TrackQuality_module.cc))

This README covers:
* where an analyzer can find things they might like to know (e.g. the definition of high quality and low quality),
* instructions for those interested in retraining or improving on the model, and
* a table of commits for each training version

## For the Interested Analyzer
The [jupyter notebook](TrkQualTrain.ipynb) contains lots of information close to the top of the file includings:
* EventNtuple datasets used (search for ```training_dataset```),
* definitions of high quality and low quality (search for ```high_qual``` and ```low_qual```),
* the features trained on (search for ```feature```), and
* the model structure (search for "Model Definitions")

## For the Interested (Re)Trainer
For those who are interested in either (a) retraining the current algorithm (e.g. we have updated reconstruction), or (b) investigating or updating an old model

### General Overview
The [jupyter notebook](TrkQualTrain.ipynb) will create an ONNX file. This can be read by the TrackQuality ArtAnalysis module.

### General Setup
You will need to create your own fork of the repository:

* go to www.github.com/Mu2e/MLTrain and click "fork"
* then in your terminal:
```
cd /path/to/your/work/area/

# only need to do this once
git clone https://www.github.com/YourGitHubUsername/MLTrain.git
cd MLTrain/
git remote add -f mu2e https://www.github.com/Mu2e/TrkQual.git

# do these whenever you are doing new development
git fetch mu2e main # get the latest and greatest
git checkout --no-track -b your-new-branchname mu2e/main
```

### Training from the Command Line
```trkqual_train.py``` runs the same steps as the notebook with no Jupyter: it makes any missing TrkQual trees, trains the models, sets each trkqual cut at the configured low-quality rejection, and saves the models, plots, the ```*plots.root``` files and a ```summary.json``` (cuts, efficiencies, AUCs, package versions). Everything that changes between trainings lives in a config file under ```configs/```:

```
mu2einit
pyenv trkqual 1.2.0
cd MLTrain/TrkQual
./trkqual_train.py configs/v3.0.yaml --outdir out/v3.0
```

For a new training, copy the latest config, change the datasets and ```training_version```, and commit the config with the model. A training is reproducible: two runs of the same config write identical ```.onnx``` and ```.ubj``` files.

To evaluate models that are already trained (for example the files in ArtAnalysis) instead of training new ones:

```
./trkqual_train.py configs/v3.0.yaml --models-from ../../ArtAnalysis/TrkDiag/data --outdir out/check
```

### Training a Model in the Notebook
For training, you need to ssh into a mu2egpvm machine with a port forwarded, and setup the correct python environment:

```
ssh -L XXXX:localhost:XXXX username@mu2egpvmYY.fnal.gov # XXXX is any port number, and YY is the gpvm number
cd /path/to/your/work/area/
mu2einit
pyenv ana
```

You can start a jupyter notebook like so:

```
cd MLTrain/TrkQual
jupyter lab --no-browser --port=XXXX # XXXX is the same port that you forwarded when you ssh'd in
```

and copy and paste the URL to your browser to open it.

You will see a directory listing of the TrkQual directory. Click the TrkQualTrain.ipynb to open the notebook in your browser.

Make any changes that you want to make:
* if this is just a retraining with updated datasets, you can just change the dataset names in section "Common Definitions", and the ```training_version_numbers``` in "Model Definitions"
* if you want to add or remove features, you can do that in "Common Definitions" too
* if you want to add a new model, then you can do that in the cell that says ```A new model can go in this cell```
   * if you want to modify the ANN1 model (e.g. change structure, or activation functioon), then I would copy it into this new cell and call it ANN2
   * if you want to try a brandh new model (e.g. a BDT), then you may need to write a new ```save_func``` etc.

Once ready, click "Kernel->Restart & Run All". You will see a bunch of plots, including some comparisons to previous models. Your model will be saved as a ```.onnx``` file in the model/ directory along with a ```*plots.root``` file containing histograms.

The ```.onnx``` file can be copied into ArtAnalysis like so:

```
cp model/TrkQual_ANN1_v2.onnx ../ArtAnalysis/TrkDiag/data/
```

and make sure that the new .onnx file is used in the TrackQuality module. (For example, change EventNtuple/fcl/prolog.fcl)

If you modified the ANN model (e.g. added new variables), then you will need to make sure the new model is implemented correctly the TrackQuality module.

If you trained a different model, then you are entering new territory and should discuss with experts how best to implement. Either:
* we make the ```TrackQuality``` module model agnostic, or
* we write separate ```TrackQuality``` modules for different models...

## Version History

| Model | Version | Commit |
|-------|---------|--------|
| ANN1 | v2 | `034f7c3` |
| ANN1 | v1.1 |`fd008e6` (previous repo) |
| ANN1 | v1 | `3d8a9b8` (previous repo) |
