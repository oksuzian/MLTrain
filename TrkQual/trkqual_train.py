#!/usr/bin/env python3
"""Train the TrkQual models from a configuration file.

This is the command-line form of TrkQualTrain.ipynb.  It runs the same steps:

  1. make the flat TrkQual trees from the EventNtuple datasets (scripts/TrkQualTree.C),
     if they do not exist yet,
  2. extract the features and the momentum resolution of each track,
  3. balance the high- and low-quality tracks of the training dataset and train each model,
  4. run each model on the training, validation and mock datasets,
  5. set each model's trkqual cut at the configured low-quality rejection, and
  6. save the models (.onnx / .ubj), plots, a ROOT file of histograms, and summary.json.

Run it in the TrkQual python environment:

    mu2einit
    pyenv trkqual 1.2.0
    ./trkqual_train.py configs/v3.0.yaml --outdir out/v3.0

To evaluate models that are already trained (for example to check a training
against the numbers it was published with), skip the training:

    ./trkqual_train.py configs/v3.0.yaml --models-from model/ --outdir out/check
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent

MOM_RES_BINS, MOM_RES_RANGE = 100, (-10, 10)
RECO_MOM_BINS, RECO_MOM_RANGE = 90, (75, 130)
# x ranges for the feature-vs-momentum-resolution plots
FEATURE_CORR_RANGES = {"nactive": (0, 100), "factive": (0.9, 1), "t0err": (0.2, 0.6), "fambig": (0, 1),
                       "fitcon": (0, 1), "momerr": (0, 0.4), "fstraws": (0.5, 2)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Dataset:
    """One TrkQual tree: the features, momenta and model predictions of its tracks."""

    def __init__(self, name, dataset, tree_dirname, training_version, is_training):
        self.name = name
        self.dataset = dataset
        self.filelist = HERE / "filelists" / f"{dataset}.list"
        self.trkqualtree = Path(tree_dirname) / f"trkqual_tree_v{training_version}_{name}.root"
        self.is_training = is_training
        self.predictions = {}

    def title(self):
        return ".".join(self.dataset.split(".")[2:4])


# ----------------------------------------------------------------------------------------------
# TrkQual trees


def make_trees(cfg, datasets):
    """Make any missing TrkQual tree with scripts/TrkQualTree.C, in the tree Musing's environment."""
    for ds in datasets:
        if ds.trkqualtree.exists():
            log(f"TrkQual tree {ds.trkqualtree} already exists")
            continue
        log(f"TrkQual tree {ds.trkqualtree} does not exist. Making it now...")
        ds.filelist.parent.mkdir(exist_ok=True)
        ds.trkqualtree.parent.mkdir(parents=True, exist_ok=True)
        script = f"""
            set -e
            # Do not mix the TrkQual python environment's ROOT libraries with the Musing's.
            unset LD_LIBRARY_PATH
            source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
            muse setup {cfg['tree_musing']}
            if [ ! -s {ds.filelist} ]; then
                setup mu2efiletools
                mu2eDatasetFileList {ds.dataset} > {ds.filelist}.tmp
                mv {ds.filelist}.tmp {ds.filelist}
            fi
            cd {HERE}
            root -l -b -q 'scripts/TrkQualTree.C++("{ds.filelist}","{ds.trkqualtree}")'
        """
        subprocess.run(["bash", "-c", script], check=True)
        if not ds.trkqualtree.exists():
            sys.exit(f"trkqual_train: scripts/TrkQualTree.C did not write {ds.trkqualtree}")


# ----------------------------------------------------------------------------------------------
# Feature extraction


def magnitude(tree_arrays, prefix):
    """|p| of a ROOT::Math vector branch, in float32 as the notebook computed it.

    np.power rather than ** : numpy's ** operator takes a square/sqrt fast path that differs from
    pow() in the last bit, which moves a few tracks across the quality boundaries.  The notebook's
    awkward arrays called np.power, so this keeps the classification identical to it.
    """
    x, y, z = (tree_arrays[f"{prefix}.fCoordinates.f{c}"] for c in "XYZ")
    return np.power(np.power(x, 2) + np.power(y, 2) + np.power(z, 2), 0.5)


def extract(cfg, ds):
    """Fill ds with the features, reco and MC momentum at the tracker entrance of its selected tracks."""
    import uproot

    derived = cfg.get("derived_features", {})
    feature_branches = [f["branch"] for f in cfg["features"]]
    needed = {"trk.status", "trk.goodfit", "trk_ent_pars.t0err"}
    for branch in feature_branches:
        needed.update(derived.get(branch, [branch]))
    for prefix in ("trk_ent.mom", "trk_ent_mc.mom"):
        needed.update(f"{prefix}.fCoordinates.f{c}" for c in "XYZ")
    if ds.is_training:
        needed.add("trk_sim.startCode")

    tree = uproot.open(f"{ds.trkqualtree}:trkqualtree")
    log(f"{ds.trkqualtree.name}: {tree.num_entries} entries")
    ds.n_entries = tree.num_entries

    features, reco_mom, mc_mom = [], [], []
    for batch in tree.iterate(sorted(needed), step_size="200 MB", library="np"):
        mask = (batch["trk.status"] > 0) & (batch["trk.goodfit"] == 1) & ~np.isnan(batch["trk_ent_pars.t0err"])
        if ds.is_training:
            mask &= batch["trk_sim.startCode"] == cfg["training_start_code"]
        with np.errstate(divide="ignore", invalid="ignore"):
            for name, (num, den) in derived.items():
                batch[name] = batch[num] / batch[den]
        features.append(np.column_stack([batch[b][mask].astype(np.float64) for b in feature_branches]))
        reco_mom.append(magnitude(batch, "trk_ent.mom")[mask].astype(np.float64))
        mc_mom.append(magnitude(batch, "trk_ent_mc.mom")[mask].astype(np.float64))

    ds.features = np.concatenate(features)
    ds.reco_mom = np.concatenate(reco_mom)
    ds.mc_mom = np.concatenate(mc_mom)
    ds.mom_res = ds.reco_mom - ds.mc_mom
    hq, lq = cfg["high_quality"], cfg["low_quality"]
    ds.high_qual = (ds.mom_res > hq["min"]) & (ds.mom_res < hq["max"])
    ds.low_qual = ds.mom_res > lq["min"]
    log(f"  {ds.name}: {len(ds.mom_res)} selected tracks, {ds.high_qual.sum()} high quality, "
        f"{ds.low_qual.sum()} low quality")


def balanced_split(cfg, training):
    """Equal numbers of high- and low-quality training tracks, split into train and test halves."""
    from sklearn.model_selection import train_test_split

    n = min(training.high_qual.sum(), training.low_qual.sum())
    x_high = training.features[training.high_qual][:n]
    x_low = training.features[training.low_qual][:n]
    x = np.concatenate((x_high, x_low))
    y = np.concatenate((np.ones(len(x_high)), np.zeros(len(x_low))))  # 1 = high quality
    split = train_test_split(x, y, test_size=cfg["test_size"], random_state=cfg["split_seed"])
    log(f"N train = {len(split[0])}, N test = {len(split[1])}")
    return split


# ----------------------------------------------------------------------------------------------
# Models


class KerasANN:
    """A fully connected network with sigmoid activations, saved as ONNX."""

    suffix = "onnx"

    def __init__(self, spec, n_features):
        self.spec = spec
        self.n_features = n_features
        self.model = None

    def fit(self, x_train, y_train, x_test, y_test):
        import keras
        import tensorflow as tf

        # Seeding before the model is built makes the initial weights, and so the training,
        # reproducible. The notebook seeded only before fit().
        keras.utils.set_random_seed(self.spec["seed"])
        tf.config.experimental.enable_op_determinism()
        model = tf.keras.Sequential(name="sequential")
        model.add(tf.keras.layers.Input(shape=(self.n_features,), batch_size=1, name="input"))
        for i, width in enumerate(self.spec["hidden_layers"]):
            model.add(tf.keras.layers.Dense(width, activation="sigmoid", name="dense" if i == 0 else f"dense_{i}"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid", name=f"dense_{len(self.spec['hidden_layers'])}"))
        model.compile(loss="binary_crossentropy", metrics=["accuracy"],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.spec["learning_rate"]))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.spec["early_stopping_patience"])
        keras.utils.set_random_seed(self.spec["seed"])
        history = model.fit(x_train, y_train, epochs=self.spec["epochs"], verbose=0,
                            validation_data=(x_test, y_test), callbacks=[early_stop])
        self.model = model
        return {"epochs_run": len(history.history["loss"])}

    def predict(self, features):
        if self.model is None:  # loaded from ONNX
            return self.onnx.run(None, {self.onnx_input: features.astype(np.float32)})[0][:, 0]
        return self.model.predict(features, verbose=0)[:, 0]

    def save(self, path):
        import onnx
        import tensorflow as tf
        import tf2onnx

        tspecs = [tf.TensorSpec(i.shape, dtype=i.dtype, name=i.name) for i in self.model.inputs]
        onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature=tspecs)
        onnx.save(onnx_model, str(path))

    def load(self, path):
        import onnx
        from onnx.reference import ReferenceEvaluator

        model = onnx.load(str(path))
        self.onnx = ReferenceEvaluator(model)
        self.onnx_input = model.graph.input[0].name


class XGBoostBDT:
    """An XGBoost boosted decision tree, saved as UBJSON."""

    suffix = "ubj"

    def __init__(self, spec, n_features):
        self.spec = spec
        self.model = None

    def fit(self, x_train, y_train, x_test, y_test):
        import xgboost as xgb

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)
        self.model = xgb.train(self.spec["params"], dtrain, num_boost_round=self.spec["num_boost_round"],
                               evals=[(dtrain, "train"), (dtest, "test")], verbose_eval=False)
        return {"trees": self.model.num_boosted_rounds()}

    def predict(self, features):
        import xgboost as xgb

        return self.model.predict(xgb.DMatrix(features))

    def save(self, path):
        self.model.save_model(str(path))

    def load(self, path):
        import xgboost as xgb

        self.model = xgb.Booster(model_file=str(path))


MODEL_TYPES = {"keras_ann": KerasANN, "xgboost_bdt": XGBoostBDT}


def trkqual_cut(cfg, training, predictions):
    """The ROC curve on the training dataset's high/low-quality tracks, and the cut at the target rejection."""
    from sklearn.metrics import auc, roc_curve

    labels = np.concatenate((np.ones(training.high_qual.sum()), np.zeros(training.low_qual.sum())))
    scores = np.concatenate((predictions[training.high_qual], predictions[training.low_qual]))
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    # the last ROC point that still rejects at least the target fraction of low-quality tracks
    cut_index = 0
    for i, f in enumerate(fpr):
        if 1 - f < cfg["rejection"]:
            cut_index = i - 1
            break
    return {"auc": float(auc(fpr, tpr)), "cut": float(thresholds[cut_index]),
            "rejection": float(1 - fpr[cut_index]), "efficiency": float(tpr[cut_index]),
            "fpr": fpr, "tpr": tpr, "thresholds": thresholds, "cut_index": cut_index}


# ----------------------------------------------------------------------------------------------
# Outputs


def write_plots(cfg, datasets, models, rocs, plotdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plotdir.mkdir(parents=True, exist_ok=True)
    training = datasets[0]
    features = cfg["features"]

    fig, ax = plt.subplots(1, 1)
    _, bins, _ = ax.hist(training.mom_res, bins=MOM_RES_BINS, range=MOM_RES_RANGE, log=True, histtype="step",
                         color="black", label="all tracks")
    ax.hist(training.mom_res[training.high_qual], bins=bins, log=True, histtype="step", label="high-quality tracks")
    ax.hist(training.mom_res[training.low_qual], bins=bins, log=True, histtype="step", label="low-quality tracks")
    ax.legend(), ax.margins(x=0), ax.grid(True)
    ax.set(title=training.title(), xlabel="Momentum Resolution [MeV/c]", ylabel="Number of Tracks")
    fig.savefig(plotdir / f"MomRes_{training.name}.png")

    fig, axs = plt.subplots(3, 3, figsize=(16, 9))
    fig.subplots_adjust(hspace=.5)
    fig.suptitle(training.title())
    for i, (f, ax) in enumerate(zip(features, axs.flatten())):
        for sel, kw in ((slice(None), {"color": "black", "label": "all tracks"}),
                        (training.high_qual, {"label": "high-quality tracks"}),
                        (training.low_qual, {"label": "low-quality tracks"})):
            ax.hist(training.features[:, i][sel], bins=100, range=f["range"], histtype="step", log=i > 0,
                    density=True, **kw)
        ax.set_xlabel(f"{f['name']} {f['unit']}"), ax.margins(0), ax.legend()
    fig.savefig(plotdir / f"Features_{training.name}.png")

    fig, axs = plt.subplots(3, 3, figsize=(16, 9))
    fig.subplots_adjust(hspace=.5)
    fig.suptitle(training.title())
    for i, (f, ax) in enumerate(zip(features, axs.flatten())):
        ax.hist2d(x=training.features[:, i], y=training.mom_res, bins=100,
                  range=[FEATURE_CORR_RANGES.get(f["name"], f["range"]), [-10, 10]], norm=LogNorm())
        ax.set_xlabel(f"{f['name']} {f['unit']}"), ax.set_ylabel("Momentum Resolution [MeV/c]"), ax.margins(0)
    fig.savefig(plotdir / f"FeatureMomCorr_{training.name}.png")

    fig, axs = plt.subplots(1, 2, figsize=(16, 9), width_ratios=[2, 1])
    axs[0].plot([0, 1], [1, 0], "k--")
    for name, roc in rocs.items():
        i = roc["cut_index"]
        for ax in axs:
            ax.plot(roc["tpr"], 1 - roc["fpr"], label=f"{name} (AUC = {roc['auc']:.3f})")
            ax.plot(roc["tpr"][i], 1 - roc["fpr"][i], "o", color=ax.lines[-1].get_color(),
                    label=f"{cfg['rejection']:.0%} rejection point (trkqual > {roc['cut']:.4f}, "
                          f"eff = {roc['efficiency']:.1%})")
    for ax in axs:
        ax.set(xlabel="High-quality track efficency (true positive rate)",
               ylabel="Low-quality track rejection (1 - false positive rate)", title="ROC curve")
        ax.legend(loc="best"), ax.grid(True)
    axs[1].set(title="ROC curve (zoom in)", ylim=(0.9, 1.0), xlim=(0, 0.8))
    fig.savefig(plotdir / f"ROCCurve_{training.name}.png")

    fig, axs = plt.subplots(1, 3, figsize=(16, 9))
    for ds, ax in zip(datasets, axs.ravel()):
        ax.hist(ds.mom_res, bins=MOM_RES_BINS, range=MOM_RES_RANGE, log=True, histtype="step", color="black",
                label="all tracks")
        for name in models:
            passed = ds.predictions[name] >= rocs[name]["cut"]
            ax.hist(ds.mom_res[passed], bins=MOM_RES_BINS, range=MOM_RES_RANGE, log=True, histtype="step",
                    label=f"{name} (eff = {ds.efficiency[name]:.1%})")
        ax.legend(), ax.margins(x=0), ax.grid(True)
        ax.set(title=ds.name, xlabel="Momentum Resolution [MeV/c]", ylabel="Number of Tracks")
    fig.savefig(plotdir / "MomRes_Comparison.png")

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for i, ds in enumerate(datasets):
        all_counts, bins, _ = axs[0][i].hist(ds.reco_mom, bins=RECO_MOM_BINS, range=RECO_MOM_RANGE, log=True,
                                             histtype="step", color="black", label="all tracks")
        for name in models:
            passed = ds.predictions[name] >= rocs[name]["cut"]
            label = f"{name} (eff = {ds.efficiency[name]:.1%})"
            cut_counts, bins, _ = axs[0][i].hist(ds.reco_mom[passed], bins=RECO_MOM_BINS, range=RECO_MOM_RANGE,
                                                 log=True, histtype="step", label=label)
            ratio = np.divide(cut_counts, all_counts, out=np.zeros_like(cut_counts), where=all_counts != 0)
            axs[1][i].errorbar((bins[:-1] + bins[1:]) / 2, ratio, xerr=(bins[1] - bins[0]) / 2, label=label)
        for row, ylabel in ((0, "Number of Tracks"), (1, "Ratio")):
            ax = axs[row][i]
            ax.legend(), ax.margins(x=0), ax.grid(True)
            ax.set(title=ds.name, xlabel="Momentum [MeV/c]", ylabel=ylabel)
        axs[1][i].set_ylim(0, 0.8)
    fig.savefig(plotdir / "MomResVsMom_Comparison.png")
    plt.close("all")


def write_histograms(cfg, datasets, name, roc, path):
    """The bookkeeping ROOT file of the notebook, with the same object names (read by scripts/CompareTrainings.C)."""
    import pandas as pd
    import uproot

    cut = roc["cut"]
    with uproot.recreate(str(path)) as out:
        for ds in datasets:
            passed = ds.predictions[name] >= cut
            out[f"all_mom_res_{ds.name}"] = np.histogram(ds.mom_res, bins=MOM_RES_BINS, range=MOM_RES_RANGE)
            out[f"high_qual_mom_{ds.name}"] = np.histogram(ds.mom_res[ds.high_qual], bins=MOM_RES_BINS,
                                                           range=MOM_RES_RANGE)
            out[f"low_qual_mom_res_{ds.name}"] = np.histogram(ds.mom_res[ds.low_qual], bins=MOM_RES_BINS,
                                                              range=MOM_RES_RANGE)
            out[f"pass_mom_res_{ds.name}"] = np.histogram(ds.mom_res[passed], bins=MOM_RES_BINS, range=MOM_RES_RANGE)
            out[f"fail_mom_res_{ds.name}"] = np.histogram(ds.mom_res[~passed], bins=MOM_RES_BINS, range=MOM_RES_RANGE)
            selections = {"all": slice(None), "high_qual": ds.high_qual, "low_qual": ds.low_qual,
                          "pass": passed, "fail": ~passed}
            for prefix, sel in selections.items():
                for i, f in enumerate(cfg["features"]):
                    values = ds.features[:, i][sel]
                    key = f"{prefix}_feature{i}_{f['name']}_{ds.name}"
                    out[key] = np.histogram(values, bins=100, range=f["range"])
                    out[f"{key}_norm"] = np.histogram(values, bins=100, range=f["range"], density=True)
        out["roc_curve"] = pd.DataFrame({"tpr": roc["tpr"], "fpr": roc["fpr"], "thresh": roc["thresholds"]})


def environment():
    versions = {"python": platform.python_version()}
    for module in ("numpy", "uproot", "sklearn", "xgboost", "tensorflow", "keras", "tf2onnx", "onnx"):
        try:
            versions[module] = __import__(module).__version__
        except ImportError:
            versions[module] = None
    commit = subprocess.run(["git", "-C", str(HERE), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    return {"versions": versions, "mltrain_commit": commit or None, "host": platform.node(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


# ----------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="training configuration (YAML), e.g. configs/v3.0.yaml")
    parser.add_argument("--outdir", default=".", help="where model/, plots/, the ROOT files and summary.json go "
                        "(default: the current directory)")
    parser.add_argument("--models-from", metavar="DIR",
                        help="do not train: evaluate the TrkQual_<model>_v<version>.<onnx|ubj> files in DIR")
    parser.add_argument("--trees-only", action="store_true", help="only make the missing TrkQual trees")
    parser.add_argument("--no-plots", action="store_true", help="skip the PNG plots")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    version = cfg["training_version"]
    outdir = Path(args.outdir)

    datasets = [Dataset(name, dataset, cfg["trkqual_tree_dirname"], version, name == "training")
                for name, dataset in cfg["datasets"].items()]
    if datasets[0].name != "training":
        sys.exit("trkqual_train: the first dataset in the config must be 'training'")
    make_trees(cfg, datasets)
    if args.trees_only:
        return

    for ds in datasets:
        extract(cfg, ds)
    training = datasets[0]

    models = {}
    for spec in cfg["models"]:
        models[spec["name"]] = MODEL_TYPES[spec["type"]](spec, len(cfg["features"]))

    fit_info = {}
    if args.models_from:
        for name, model in models.items():
            path = Path(args.models_from) / f"TrkQual_{name}_v{version}.{model.suffix}"
            log(f"Loading {name} from {path}")
            model.load(path)
            fit_info[name] = {"loaded_from": str(path.resolve())}
    else:
        x_train, x_test, y_train, y_test = balanced_split(cfg, training)
        for name, model in models.items():
            log(f"Training {name}...")
            start = time.time()
            fit_info[name] = model.fit(x_train, y_train, x_test, y_test)
            fit_info[name]["seconds"] = round(time.time() - start, 1)
            log(f"  done in {fit_info[name]['seconds']} s")

    for ds in datasets:
        for name, model in models.items():
            ds.predictions[name] = model.predict(ds.features)

    rocs = {}
    for name in models:
        rocs[name] = trkqual_cut(cfg, training, training.predictions[name])
        r = rocs[name]
        log(f"An {name} trkqual cut of {round(r['cut'], 4)} has a low-quality track rejection of "
            f"{round(r['rejection'] * 100, 1)}% with a high-quality track efficiency of {round(r['efficiency'] * 100, 1)}%")
    for ds in datasets:
        ds.efficiency = {name: float(((ds.predictions[name] >= rocs[name]["cut"]) & ds.high_qual).sum()
                                     / ds.high_qual.sum()) for name in models}

    if not args.models_from:
        (outdir / "model").mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            path = outdir / "model" / f"TrkQual_{name}_v{version}.{model.suffix}"
            model.save(path)
            log(f"Saved {path}")
    outdir.mkdir(parents=True, exist_ok=True)
    for name in models:
        write_histograms(cfg, datasets, name, rocs[name], outdir / f"TrkQual_{name}_v{version}_plots.root")
    if not args.no_plots:
        write_plots(cfg, datasets, models, rocs, outdir / "plots" / f"v{version}")

    summary = {
        "config": cfg,
        "environment": environment(),
        "datasets": {ds.name: {"dataset": ds.dataset, "trkqual_tree": str(ds.trkqualtree), "entries": ds.n_entries,
                               "selected": int(len(ds.mom_res)), "high_quality": int(ds.high_qual.sum()),
                               "low_quality": int(ds.low_qual.sum())} for ds in datasets},
        "models": {name: {**fit_info[name],
                          "auc": rocs[name]["auc"], "trkqual_cut": rocs[name]["cut"],
                          "rejection": rocs[name]["rejection"],
                          "efficiency": {ds.name: ds.efficiency[name] for ds in datasets}} for name in models},
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Wrote {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
