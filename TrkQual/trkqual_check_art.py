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
     against art's score for the same track.  With --summary, also the high-quality track
     efficiency at each model's trkqual cut, from art's scores and from the training's.

It exits non-zero if any track disagrees.  Make the EventNtuples first (see README.md), then:

    ./trkqual_check_art.py configs/v3.0.yaml --ntuples nts.list \\
        --leaf trkqual_ann_v3_0=out/v3.0/model/TrkQual_ANN1_v3.0.onnx \\
        --leaf trkqual_bdt_v3_0=out/v3.0/model/TrkQual_BDT1_v3.0.ubj \\
        --summary out/v3.0/summary.json --workdir out/v3.0/check_art
"""

import argparse
import json
import re
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
        models[branch] = model
    return models


def cuts_from_summary(summary_path, models):
    """{branch: trkqual cut} for the models whose file names match a model of the summary's training."""
    summary = json.load(open(summary_path))
    version = summary["config"]["training_version"]
    cuts = {}
    for branch, model in models.items():
        m = re.fullmatch(r"TrkQual_(\w+)_v(.+)", model.path.stem)
        if m and m.group(2) == version and m.group(1) in summary["models"]:
            cuts[branch] = summary["models"][m.group(1)]["trkqual_cut"]
    if not cuts:
        sys.exit(f"trkqual_check_art: no --leaf model file is a model of the v{version} training in {summary_path}")
    tt.log(f"Cuts from {summary_path}: " + ", ".join(f"{b} > {c:.4f}" for b, c in cuts.items()))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="the training configuration (YAML), e.g. configs/v3.0.yaml")
    parser.add_argument("--ntuples", required=True, metavar="LIST",
                        help="text file listing the EventNtuple ROOT files made in art, one per line")
    parser.add_argument("--leaf", action="append", required=True, metavar="BRANCH=MODELFILE",
                        help="an EventNtuple trkqual branch and the model file art used for it (repeatable), "
                             "e.g. trkqual_bdt_v3_0=out/v3.0/model/TrkQual_BDT1_v3.0.ubj")
    parser.add_argument("--summary", help="summary.json of the training: also compare efficiencies at its cuts")
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
    cuts = cuts_from_summary(args.summary, models) if args.summary else {}
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
    high_qual = ds.high_qual[matched]
    for branch, cut in cuts.items():
        passed_art, passed_python = art_b[branch] >= cut, python_b[branch] >= cut
        efficiency[branch] = {"cut": cut, "high_quality_tracks": int(high_qual.sum()),
                              "art": float((passed_art & high_qual).sum() / high_qual.sum()),
                              "training": float((passed_python & high_qual).sum() / high_qual.sum()),
                              "tracks_passing_in_one_only": int((passed_art != passed_python).sum())}
        e = efficiency[branch]
        tt.log(f"  {branch}: high-quality efficiency at trkqual > {cut:.4f}: art {e['art']:.2%}, training "
               f"{e['training']:.2%}; {e['tracks_passing_in_one_only']} tracks pass in one and fail in the other")
        if e["tracks_passing_in_one_only"]:
            failures.append(f"efficiency {branch}")

    out = workdir / "check_art.json"
    with open(out, "w") as f:
        json.dump({"config": args.config, "ntuples": [str(f) for f in files], "tolerance": args.tolerance,
                   "models": {b: str(m.path.resolve()) for b, m in models.items()},
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
