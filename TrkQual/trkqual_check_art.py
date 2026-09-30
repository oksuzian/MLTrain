#!/usr/bin/env python3
"""Check, track by track, that TrkQual in art gives the scores of the training.

The TrackQuality module (ArtAnalysis/TrkDiag/src/TrackQuality_module.cc) builds the features from
the KalSeed and runs the models with ONNXRuntime and the XGBoost C API.  The training builds them
from EventNtuple branches (scripts/TrkQualTree.C) and runs the models in python.  On EventNtuples
made with the models under test, this script checks that the two agree:

  A. every track: the features built from the EventNtuple the way the module builds them, scored
     in python, against the score art wrote;
  B. the tracks the training would select (scripts/TrkQualTree.C, then the cuts of
     trkqual_train.py): the features built the way the training builds them, scored in python,
     against art's score for the same track.  For each model with a trkqual cut (from a
     trkqual_train.py summary.json that records the same model file, or --cut), also the
     high-quality track efficiency at the cut, from art's scores and from the training's.

It exits non-zero if any track disagrees.  It does not run art: make the EventNtuples first (see
README.md), then:

    ./trkqual_check_art.py configs/v3.0.yaml --ntuples nts.list \\
        --leaf trkqual_ann_v3_0=out/v3.0/model/TrkQual_ANN1_v3.0.onnx \\
        --leaf trkqual_bdt_v3_0=out/v3.0/model/TrkQual_BDT1_v3.0.ubj \\
        --summary out/v3.0/summary.json --workdir out/v3.0/check_art

With --plots, it also plots the models against each other on the selected tracks, from art's
scores: score distributions, ROC curves, and the efficiency against momentum and the momentum
resolution of the tracks passing each cut.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

import trkqual_train as tt

# The features in the order TrackQuality_module.cc fills them.  The module hard-codes them, so a
# config with other features cannot be checked against it.
MODULE_FEATURES = ["nactive", "factive", "t0err", "fambig", "fitcon", "momerr", "fstraws"]
TT_FRONT = 0  # SurfaceIdDetail::TT_Front, the tracker entrance
NO_ENTRANCE = -9999.0  # the module's t0err and momerr for a track with no tracker-entrance intersection

# The branches that identify a track in both the EventNtuple and the TrkQual tree
MATCH_FIELDS = ["run", "subrun", "event", "nhits", "nactive", "nnullambig", "nmatactive", "fitcon"]
NTUPLE_BRANCH = {"run": "evtinfo/run", "subrun": "evtinfo/subrun", "event": "evtinfo/event",
                 **{f: f"trk/trk.{f}" for f in MATCH_FIELDS[3:]}}
TREE_BRANCH = {"run": "evtinfo.run", "subrun": "evtinfo.subrun", "event": "evtinfo.event",
               **{f: f"trk.{f}" for f in MATCH_FIELDS[3:]}}


def load_models(leaves):
    """{branch: model} from the --leaf BRANCH=MODELFILE arguments; the model type from the suffix."""
    types = {cls.suffix: cls for cls in tt.MODEL_TYPES.values()}
    models = {}
    for leaf in leaves:
        branch, sep, path = leaf.partition("=")
        path = Path(path)
        if not sep or not branch or path.suffix.lstrip(".") not in types:
            sys.exit(f"trkqual_check_art: --leaf {leaf}: expected BRANCH=MODELFILE with a "
                     f"{' or '.join('.' + s for s in types)} model file")
        if not path.exists():
            sys.exit(f"trkqual_check_art: {path} does not exist")
        model = types[path.suffix.lstrip(".")]({}, len(MODULE_FEATURES))
        model.load(path)
        model.path = path
        model.sha256 = tt.file_record(path)["sha256"]
        models[branch] = model
    return models


def find_cuts(summaries, explicit, models):
    """{branch: trkqual cut}: from --cut, else from the summary.json that records the same model file.

    A summary records the SHA-256 of each model file it trained or evaluated, so the cut is found
    from the file's content, wherever the file was copied to.
    """
    cuts = {}
    for arg in explicit:
        branch, sep, value = arg.partition("=")
        if not sep or branch not in models:
            sys.exit(f"trkqual_check_art: --cut {arg}: expected BRANCH=VALUE with BRANCH one of {list(models)}")
        cuts[branch] = float(value)
        tt.log(f"Cut for {branch} from --cut: {cuts[branch]:.4f}")
    for path in summaries:
        recorded = json.load(open(path))["models"]
        if not all("sha256" in m for m in recorded.values()):
            sys.exit(f"trkqual_check_art: {path} does not record its model files; remake it with trkqual_train.py")
        by_sha = {m["sha256"]: (name, m["trkqual_cut"]) for name, m in recorded.items()}
        found = [b for b, model in models.items() if model.sha256 in by_sha]
        if not found:
            sys.exit(f"trkqual_check_art: none of the --leaf model files is a model of {path}")
        for branch in found:
            if branch not in cuts:
                name, cuts[branch] = by_sha[models[branch].sha256]
                tt.log(f"Cut for {branch} from {path} ({name}): {cuts[branch]:.4f}")
    for branch in models:
        if branch not in cuts and (summaries or explicit):
            tt.log(f"No cut for {branch}: no --cut, and no --summary records its model file")
    return cuts


def score(models, features):
    """python scores of the feature rows, in float32 as the module feeds them"""
    f32 = features.astype(np.float32)
    return {branch: np.asarray(model.predict(f32), dtype=np.float32) for branch, model in models.items()}


def compare(art, python, tolerance):
    """per model: the number of tracks, of tracks whose scores differ by more than tolerance, and the largest difference"""
    result = {}
    for branch in art:
        a, p = art[branch].astype(np.float64), python[branch].astype(np.float64)
        diff = np.where(np.isnan(a) & np.isnan(p), 0.0, np.abs(a - p))
        result[branch] = {"tracks": int(len(a)), "differ": int((~(diff <= tolerance)).sum()),  # NaN in one only differs
                          "max_abs_diff": float(np.nanmax(diff)) if len(a) else 0.0}
    return result


def report(title, result, tolerance):
    tt.log(title)
    for branch, r in result.items():
        tt.log(f"  {branch}: {r['tracks']} tracks, {r['differ']} differ by more than {tolerance:g} "
               f"(largest difference {r['max_abs_diff']:.1e})")


def read_ntuples(files, branches):
    """Every track of the EventNtuples: its identifying fields, the features as the module builds them, and art's scores."""
    import awkward as ak
    import uproot

    cols = defaultdict(list)
    for f in files:
        a = uproot.open(f"{f}:EventNtuple/ntuple").arrays(
            list(NTUPLE_BRANCH.values()) + ["trksegs", "trksegpars_lh"]
            + [f"{b}/{b}.result" for b in branches], how=dict)
        ntrk = ak.num(a["trk/trk.nhits"])
        for b in branches:
            if not ak.all(ak.num(a[f"{b}/{b}.result"]) == ntrk):
                sys.exit(f"trkqual_check_art: {f}: {b} does not have one entry per track")
            cols[b].append(ak.to_numpy(ak.flatten(a[f"{b}/{b}.result"])))
        for field in ("run", "subrun", "event"):
            cols[field].append(np.repeat(ak.to_numpy(a[NTUPLE_BRANCH[field]]), ak.to_numpy(ntrk)))
        for field in MATCH_FIELDS[3:]:
            cols[field].append(ak.to_numpy(ak.flatten(a[NTUPLE_BRANCH[field]])))
        # the module takes t0err and momerr from the first tracker-entrance intersection
        entrance = a["trksegs"].sid == TT_FRONT
        first = ak.argmax(entrance, axis=2, keepdims=True)
        cols["has_entrance"].append(ak.to_numpy(ak.flatten(ak.any(entrance, axis=2))))
        for name, values in (("t0err", a["trksegpars_lh"].t0err), ("momerr", a["trksegs"].momerr)):
            cols[name].append(ak.to_numpy(ak.flatten(ak.fill_none(ak.firsts(values[first], axis=2), np.nan))))
        tt.log(f"{Path(f).name}: {len(ntrk)} events, {int(ak.sum(ntrk))} tracks")
    return {k: np.concatenate(v) for k, v in cols.items()}


def module_features(tracks):
    """The features as TrackQuality_module.cc builds them"""
    has_entrance = tracks["has_entrance"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack([
            tracks["nactive"].astype(np.float64),
            tracks["nactive"] / tracks["nhits"],
            np.where(has_entrance, tracks["t0err"], NO_ENTRANCE),
            tracks["nnullambig"] / tracks["nactive"],
            tracks["fitcon"].astype(np.float64),
            np.where(has_entrance, tracks["momerr"], NO_ENTRANCE),
            tracks["nmatactive"] / tracks["nactive"],
        ])


def match_tracks(tracks, selected):
    """For each track of the training's selection, the index of the same track in the EventNtuples (-1 if not unique)."""
    index = defaultdict(list)
    for i, key in enumerate(zip(*(tracks[f].tolist() for f in MATCH_FIELDS))):
        index[key].append(i)
    match = np.full(len(selected[MATCH_FIELDS[0]]), -1)
    ambiguous = missing = 0
    for j, key in enumerate(zip(*(selected[f].tolist() for f in MATCH_FIELDS))):
        found = index.get(key, [])
        if len(found) == 1:
            match[j] = found[0]
        elif found:
            ambiguous += 1
        else:
            missing += 1
    return match, ambiguous, missing


def write_plots(plotdir, title, models, scores, cuts, high_qual, low_qual, reco_mom, mom_res):
    """The models against each other on the selected tracks, from art's scores. Returns the AUC of each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from sklearn.metrics import auc, roc_curve

    plotdir.mkdir(parents=True, exist_ok=True)
    color = {b: f"C{i}" for i, b in enumerate(models)}
    groups = defaultdict(list)  # one panel per kind of model
    for b, m in models.items():
        groups[f".{m.suffix} models"].append(b)
    subtitle = f"{title}\n{int(high_qual.sum())} high-quality and {int(low_qual.sum())} low-quality tracks"

    def passes(b):
        return scores[b] >= cuts[b]

    def panels(n_rows=1, **kw):
        fig, axs = plt.subplots(n_rows, len(groups), figsize=(7 * len(groups), 6 * n_rows), squeeze=False, **kw)
        return fig, axs

    fig, axs = panels()
    bins = np.linspace(0, 1, 51)
    for ax, (group, branches) in zip(axs[0], groups.items()):
        for b in branches:
            ax.hist(scores[b][high_qual], bins=bins, histtype="step", density=True, color=color[b], label=f"{b}, high quality")
            ax.hist(scores[b][low_qual], bins=bins, histtype="step", density=True, color=color[b], ls="--",
                    label=f"{b}, low quality")
            if b in cuts:
                ax.axvline(cuts[b], color=color[b], ls=":", label=f"{b} cut {cuts[b]:.4f}")
        ax.set(yscale="log", xlabel="trkqual", ylabel="tracks (normalised)", title=group)
        ax.legend(loc="upper center", fontsize=8), ax.margins(x=0), ax.grid(True, alpha=0.3)
    fig.suptitle(subtitle)
    fig.savefig(plotdir / "scores.png", dpi=110, bbox_inches="tight")

    aucs = {}
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    labels = np.concatenate((np.ones(high_qual.sum()), np.zeros(low_qual.sum())))
    zoom_tpr = []
    for b in models:
        fpr, tpr, _ = roc_curve(labels, np.concatenate((scores[b][high_qual], scores[b][low_qual])))
        aucs[b] = float(auc(fpr, tpr))
        zoom_tpr.extend(tpr[1 - fpr >= 0.95])
        for ax in axs:
            ax.plot(tpr, 1 - fpr, color=color[b], label=f"{b} (AUC {aucs[b]:.3f})")
            if b in cuts:
                eff = (passes(b) & high_qual).sum() / high_qual.sum()
                rej = 1 - (passes(b) & low_qual).sum() / low_qual.sum()
                ax.plot(eff, rej, "o", color=color[b], label=f"{b} at its cut: eff {eff:.1%}, rejection {rej:.1%}")
    for ax in axs:
        ax.set(xlabel="high-quality track efficiency", ylabel="low-quality track rejection")
        ax.grid(True, alpha=0.3)
    axs[0].set_title("ROC"), axs[0].legend(loc="lower left", fontsize=8)
    axs[1].set(title="zoom", ylim=(0.95, 1.0), xlim=(min(zoom_tpr), max(zoom_tpr)))
    fig.suptitle(subtitle)
    fig.savefig(plotdir / "roc.png", dpi=110, bbox_inches="tight")

    if cuts:
        lo, hi = np.percentile(reco_mom[high_qual], [0.5, 99.5])
        mbins = np.linspace(np.floor(lo), np.ceil(hi), 26)
        centers = (mbins[1:] + mbins[:-1]) / 2
        n_all, _ = np.histogram(reco_mom[high_qual], bins=mbins)
        fig, axs = panels(2, sharex=True)
        for col, (group, branches) in enumerate(groups.items()):
            axs[0][col].stairs(n_all, mbins, color="black", label="high-quality tracks")
            for b in (b for b in branches if b in cuts):
                n_pass, _ = np.histogram(reco_mom[high_qual & passes(b)], bins=mbins)
                axs[0][col].stairs(n_pass, mbins, color=color[b], label=f"passing {b}")
                with np.errstate(divide="ignore", invalid="ignore"):
                    e = n_pass / n_all
                    err = np.sqrt(e * (1 - e) / n_all)
                axs[1][col].errorbar(centers, e, yerr=err, xerr=(mbins[1] - mbins[0]) / 2, fmt="o", ms=3,
                                     color=color[b], label=b)
            axs[0][col].set(yscale="log", ylabel="tracks", title=group)
            axs[1][col].set(xlabel="reco momentum at the tracker entrance [MeV/c]",
                            ylabel="high-quality efficiency", ylim=(0, 1))
            for ax in axs[:, col]:
                ax.legend(fontsize=8), ax.grid(True, alpha=0.3)
        fig.suptitle(subtitle)
        fig.savefig(plotdir / "eff_vs_mom.png", dpi=110, bbox_inches="tight")

        fig, axs = panels()
        rbins = np.linspace(-10, 10, 101)
        for ax, (group, branches) in zip(axs[0], groups.items()):
            ax.hist(mom_res, bins=rbins, histtype="step", color="black", label="all selected tracks")
            for b in (b for b in branches if b in cuts):
                ax.hist(mom_res[passes(b)], bins=rbins, histtype="step", color=color[b],
                        label=f"passing {b} ({int(passes(b).sum())} tracks)")
            ax.set(yscale="log", xlabel="momentum resolution, reco - MC at the tracker entrance [MeV/c]",
                   ylabel="tracks", title=group)
            ax.legend(fontsize=8), ax.margins(x=0), ax.grid(True, alpha=0.3)
        fig.suptitle(subtitle)
        fig.savefig(plotdir / "momres_pass.png", dpi=110, bbox_inches="tight")

    for branches in groups.values():  # per track: the first model of a kind against each other
        for other in branches[1:]:
            first = branches[0]
            fig, ax = plt.subplots(figsize=(8, 6.5))
            h = ax.hist2d(scores[first], scores[other], bins=100, range=[[0, 1], [0, 1]], norm=LogNorm())
            fig.colorbar(h[3], ax=ax, label="tracks")
            if first in cuts:
                ax.axvline(cuts[first], color=color[first], ls=":", label=f"{first} cut {cuts[first]:.4f}")
            if other in cuts:
                ax.axhline(cuts[other], color=color[other], ls=":", label=f"{other} cut {cuts[other]:.4f}")
            ax.set(xlabel=first, ylabel=other, title=subtitle)
            if first in cuts or other in cuts:
                ax.legend(loc="upper left", fontsize=8)
            fig.savefig(plotdir / f"scores_{first}_vs_{other}.png", dpi=110, bbox_inches="tight")
    plt.close("all")
    tt.log(f"Plots in {plotdir}")
    return aucs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="the training configuration (YAML), e.g. configs/v3.0.yaml")
    parser.add_argument("--ntuples", required=True, metavar="LIST",
                        help="text file listing the EventNtuple ROOT files made in art, one per line")
    parser.add_argument("--leaf", action="append", required=True, metavar="BRANCH=MODELFILE",
                        help="an EventNtuple trkqual branch and the model file art used for it (repeatable), "
                             "e.g. trkqual_bdt_v3_0=out/v3.0/model/TrkQual_BDT1_v3.0.ubj")
    parser.add_argument("--summary", action="append", default=[],
                        help="a trkqual_train.py summary.json (repeatable): the cut of each --leaf model file it "
                             "records, found by the file's SHA-256")
    parser.add_argument("--cut", action="append", default=[], metavar="BRANCH=VALUE",
                        help="the trkqual cut of a --leaf branch (repeatable); takes precedence over --summary")
    parser.add_argument("--plots", metavar="DIR", help="also plot the models against each other in DIR")
    parser.add_argument("--title", default="scores from art", help="the plots' title")
    parser.add_argument("--workdir", default="check_art", help="where the TrkQual tree (remade on every run) and check_art.json go")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="largest allowed score difference")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if [f["name"] for f in cfg["features"]] != MODULE_FEATURES:
        sys.exit(f"trkqual_check_art: the TrackQuality module uses the features {MODULE_FEATURES}; "
                 f"{args.config} has {[f['name'] for f in cfg['features']]}")
    files = [Path(line.strip()).resolve() for line in open(args.ntuples) if line.strip()]
    for f in files:
        if not f.exists():
            sys.exit(f"trkqual_check_art: {f} does not exist")
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    models = load_models(args.leaf)
    cuts = find_cuts(args.summary, args.cut, models)
    failures = []

    # A. the module's features, every track
    tracks = read_ntuples(files, list(models))
    features = module_features(tracks)
    if np.isinf(features).any():
        sys.exit("trkqual_check_art: a track has an infinite feature, which the module cannot score")
    python = score(models, features)
    for branch in models:
        python[branch][~tracks["has_entrance"]] = 0.0  # the module's score for a track with no tracker entrance
    art = {branch: tracks[branch] for branch in models}
    result_a = compare(art, python, args.tolerance)
    tt.log(f"{len(files)} EventNtuple files, {len(features)} tracks, "
           f"{int((~tracks['has_entrance']).sum())} with no tracker-entrance intersection")
    report("A. art against python, features as the TrackQuality module builds them, all tracks:",
           result_a, args.tolerance)
    failures += [f"A {b}" for b, r in result_a.items() if r["differ"]]

    # B. the training's features, on the tracks it would select
    version = cfg["training_version"]
    ds = tt.Dataset("check_art", "EventNtuples made in art", workdir, version, is_training=False)
    ds.filelist = workdir / "ntuples.list"
    ds.filelist.write_text("".join(f"{f}\n" for f in files))
    if ds.trkqualtree.exists():  # made from an earlier --ntuples list, so remake it
        ds.trkqualtree.unlink()
    tt.make_trees(cfg, [ds])
    tt.extract(cfg, ds, extra_branches=list(TREE_BRANCH.values()))
    selected = {field: ds.extra[TREE_BRANCH[field]] for field in MATCH_FIELDS}
    match, ambiguous, missing = match_tracks(tracks, selected)
    matched = match >= 0
    tt.log(f"{int(matched.sum())} of the {len(match)} selected tracks found in the EventNtuples "
           f"({ambiguous} not uniquely, {missing} not at all)")
    if missing:
        sys.exit(f"trkqual_check_art: {missing} tracks of {ds.trkqualtree} are not in the EventNtuples")
    rows = match[matched]
    python_b = score(models, ds.features[matched])
    art_b = {branch: art[branch][rows] for branch in models}
    result_b = compare(art_b, python_b, args.tolerance)
    report("B. art against python, features as the training builds them, tracks the training selects:",
           result_b, args.tolerance)
    failures += [f"B {b}" for b, r in result_b.items() if r["differ"]]

    differ = (module_features({k: v[rows] for k, v in tracks.items()}).astype(np.float32)
              != ds.features[matched].astype(np.float32))
    feature_differences = {name: int(differ[:, i].sum()) for i, name in enumerate(MODULE_FEATURES)}
    tt.log(f"  tracks whose module and training features differ, per feature: {feature_differences}")

    efficiency = {}
    high_qual, low_qual = ds.high_qual[matched], ds.low_qual[matched]
    for branch, cut in cuts.items():
        passed_art, passed_python = art_b[branch] >= cut, python_b[branch] >= cut
        efficiency[branch] = {"cut": cut, "high_quality_tracks": int(high_qual.sum()),
                              "art": float((passed_art & high_qual).sum() / high_qual.sum()),
                              "training": float((passed_python & high_qual).sum() / high_qual.sum()),
                              "low_quality_rejection_art": float(1 - (passed_art & low_qual).sum() / low_qual.sum()),
                              "tracks_passing_in_one_only": int((passed_art != passed_python).sum())}
        e = efficiency[branch]
        tt.log(f"  {branch}: high-quality efficiency at trkqual > {cut:.4f}: art {e['art']:.2%}, training "
               f"{e['training']:.2%}; {e['tracks_passing_in_one_only']} tracks pass in one and fail in the other; "
               f"low-quality rejection {e['low_quality_rejection_art']:.2%}")
        if e["tracks_passing_in_one_only"]:
            failures.append(f"efficiency {branch}")

    aucs = {}
    if args.plots:
        aucs = write_plots(Path(args.plots), args.title, models, art_b, cuts, high_qual, low_qual,
                           ds.reco_mom[matched], ds.mom_res[matched])

    out = workdir / "check_art.json"
    with open(out, "w") as f:
        json.dump({"config": args.config, "ntuples": [str(f) for f in files], "tolerance": args.tolerance,
                   "models": {b: {"model_file": str(m.path.resolve()), "sha256": m.sha256} for b, m in models.items()},
                   "auc_art": aucs,
                   "all_tracks": result_a, "selected_tracks": result_b,
                   "selected": {"tracks": int(len(match)), "matched": int(matched.sum()), "ambiguous": ambiguous},
                   "feature_differences": feature_differences, "efficiency": efficiency,
                   "failures": failures}, f, indent=2)
    tt.log(f"Wrote {out}")
    if failures:
        sys.exit(f"trkqual_check_art: FAILED: {', '.join(failures)}")
    tt.log("PASSED: art gives the training's scores for every track")


if __name__ == "__main__":
    main()
