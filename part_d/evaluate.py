"""Freeze normal-validation decisions, then run both methods on the same official test IDs."""
import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from metrics import PixelAUROC, align, frozen_threshold, image_auc, native_map, native_mask
from prepare_inputs import LOCK, digest

GROUPS = {"mvtec_ad": ("bottle", "cable", "metal_nut", "tile", "wood"),
          "local_capture": ("bottle", "nut")}
RATIOS = (0, 1, 5, 10)  # 0 denotes B's autoencoder, which has no coreset.
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return Path(path).read_text(encoding="utf-8-sig")


def obj(path):
    return json.loads(read(path))


def json_out(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def csv_in(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_out(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_name(ratio):
    return f"pc{ratio:02d}" if ratio else "ae"


def folders(inputs):
    return [inputs / part / "project" for part in ("A", "B", "C")]


def split(manifest, dataset, category, subset):
    return [dict(r, dataset=dataset) for r in manifest["samples"]
            if r["category"] == category and r["split"] == subset]


def read_predictions(directory):
    return [json.loads(line) for line in read(directory / "predictions.jsonl").splitlines() if line.strip()]


def map_file(folder, pred):
    path = (folder / pred["anomaly_map_path"]).resolve()
    require(path.is_relative_to(folder.resolve()), f"Map outside prediction directory: {path}")
    return path


def load_map(folder, truth, pred):
    path = map_file(folder, pred)
    field = np.load(path, allow_pickle=False)
    require(tuple(pred["original_size"]) == (truth["height"], truth["width"]), f"Native size: {truth['sample_id']}")
    require(field.dtype == np.float32 and field.shape == tuple(pred["map_shape"]) == (128, 128)
            and np.isfinite(field).all(), f"Raw anomaly map: {path}")
    return field


def pc_folder(c, dataset, category, ratio, subset):
    parent = "validation" if subset == "val" else "test"
    return c / "outputs" / parent / dataset / category / f"ratio_{ratio:02d}" / subset


def check_local_ids(manifest):
    owner = {}
    for row in manifest["samples"]:
        key = (row["category"], row["physical_id"])
        require(owner.setdefault(key, row["split"]) == row["split"], f"Physical specimen shared across splits: {key}")


def freeze(inputs, out):
    a,b,c = folders(inputs)
    target = out / "results/protocol_lock.json"
    require(not target.exists(), f"Lock already exists; reuse it or choose a fresh output directory: {target}")
    up_lock = obj(c / "protocol_lock.json")
    pc_inventory = {(r["dataset"],r["category"],r["ratio_percent"]):r for r in up_lock["results"]}
    require(up_lock["main_ratio_percent"] == 5 and up_lock["status"] == "LOCKED", "C main configuration changed")
    choices = []
    for dataset, categories in GROUPS.items():
        afile = a / "manifests" / (dataset+".json")
        bfile = b / "manifests" / (dataset+".json")
        manifest = obj(afile)
        require(manifest == obj(bfile), f"A/B parsed manifests differ for {dataset}")
        if dataset == "local_capture":
            check_local_ids(manifest)
        for cat in categories:
            val = split(manifest,dataset,cat,"val")
            for ratio in RATIOS:
                vdir = pc_folder(c,dataset,cat,ratio,"val") if ratio else b/"predictions"/dataset/cat/"val"
                cfg_path = vdir/"model_config.json"
                cfg = obj(cfg_path)
                pairs = align(val, read_predictions(vdir))
                for truth,pred in pairs:
                    load_map(vdir,truth,pred)
                threshold = frozen_threshold(pairs)
                train_model = (vdir.parent/"memory_bank.npy" if ratio else b/"artifacts"/dataset/cat/"model.pt")
                model_digest = digest(train_model)
                cfg_key = "bank_sha256" if ratio else "checkpoint_sha256"
                require(cfg[cfg_key] == model_digest and cfg["dataset"] == dataset and cfg["category"] == cat
                        and cfg["image_size"] == 128, f"Configuration mismatch: {vdir}")
                require(cfg["manifest_sha256"] == digest(afile if ratio else bfile), f"Training manifest hash: {vdir}")
                if ratio:
                    original = pc_inventory[(dataset,cat,ratio)]
                    require(digest(vdir/"predictions.jsonl") == original["predictions_sha256"]
                            and digest(cfg_path) == original["config_sha256"]
                            and model_digest == original["bank_sha256"], f"C lock mismatch: {vdir}")
                    original_threshold = original["provisional_threshold"]
                else:
                    original = obj(b/"artifacts"/dataset/cat/"threshold.json")
                    original_threshold = original["threshold"]
                require(math.isclose(threshold,original_threshold,rel_tol=1e-6,abs_tol=1e-8),
                        f"Threshold disagrees with validation export: {vdir}: {threshold} vs {original_threshold}")
                if not ratio:
                    demo = {k: cfg[k] for k in ("dataset","category","image_size","top_fraction","checkpoint_sha256")}
                    demo.update(threshold=threshold,status="locked_by_D",quantile=.99,validation_count=len(val),
                                comparison="score > threshold")
                    json_out(out/"demo"/"artifacts"/dataset/cat/"threshold.json",demo)
                choices.append(dict(dataset=dataset,category=cat,model=model_name(ratio),ratio_percent=ratio,
                                    threshold=threshold,validation_count=len(val),validation_jsonl_sha256=digest(vdir/"predictions.jsonl"),
                                    config_sha256=digest(cfg_path),model_sha256=model_digest,a_manifest_sha256=digest(afile),
                                    training_manifest_sha256=cfg["manifest_sha256"],bank_bytes=cfg.get("bank_bytes",0),
                                    bank_patches=cfg.get("bank_patches",0),val_coverage_p95=original.get("coverage_p95")))
    require(len(choices)==28 and sum(x["validation_count"] for x in choices)==1024, "Validation coverage")
    json_out(target,dict(status="locked_before_D_test",utc=datetime.now(timezone.utc).isoformat(),source_releases=LOCK,
                         source_commit=os.getenv("GITHUB_SHA"),protocol_sha256=digest(HERE/"protocol.md"),
                         c_lock_sha256=digest(c/"protocol_lock.json"),main_models=["ae","pc05"],
                         image_decision="score > linear 0.99 quantile of normal validation scores",pixel_threshold=None,
                         settings=choices))
    csv_out(out/"results/thresholds.csv", choices)
    print(f"Frozen {len(choices)} configurations on 256 unique normal validation images.",flush=True)


def run_autoencoder(b,a,out,dataset,cat,truth,setup):
    import torch
    sys.path.insert(0,str(b)) if str(b) not in sys.path else None
    from autoencoder.core import load_model, load_threshold, predict
    torch.set_num_threads(4)
    weight = b/"artifacts"/dataset/cat/"model.pt"
    net,cfg = load_model(weight)
    cfg = dict(cfg,checkpoint_sha256=digest(weight))
    threshold = load_threshold(out/"demo"/"artifacts"/dataset/cat/"threshold.json",cfg)
    require(cfg["checkpoint_sha256"] == setup["model_sha256"] and threshold["threshold"] == setup["threshold"],
            f"Autoencoder lock mismatch: {dataset}/{cat}")
    folder = out/"predictions"/"ae"/dataset/cat/"test"
    (folder/"maps").mkdir(parents=True,exist_ok=True)
    with (folder/"predictions.jsonl").open("w",encoding="utf-8") as stream:
        for i,row in enumerate(truth):
            with Image.open(a/row["image_path"]) as im:
                require(im.size==(row["width"],row["height"]), f"Image size: {row['sample_id']}")
                score,field,_ = predict(net,cfg,im)
            filename=f"maps/{i:05d}.npy"
            np.save(folder/filename,field,allow_pickle=False)
            item=dict(sample_id=row["sample_id"],dataset=dataset,category=cat,split="test",score=score,
                      anomaly_map_path=filename,map_shape=list(field.shape),original_size=[row["height"],row["width"]],
                      threshold=threshold["threshold"],is_anomaly=score>threshold["threshold"])
            stream.write(json.dumps(item,allow_nan=False)+"\n")
    json_out(folder/"model_config.json",cfg)
    return folder


def summarize_timing(c, a, out):
    src = c/"latency_benchmark"
    report = obj(src/"benchmark_report.json")
    recorded = csv_in(src/"latency_summary.csv")
    raw = csv_in(src/"latency_raw.csv")
    require(len(recorded)==28 and len(raw)==840 and report["status"]=="PASS", "Timing scope changed")
    groups=defaultdict(list)
    def key(r):
        return (r["dataset"],r["category"],int(r["ratio_percent"] or 0))
    for row in raw:
        groups[key(row)].append(row)
    output={}
    for r in recorded:
        k=key(r)
        seq=sorted(groups[k],key=lambda x:int(x["iteration"]))
        val=split(obj(a/"manifests"/(k[0]+".json")),k[0],k[1],"val")[:15]
        require([s["sample_id"] for s in seq]==[val[i%len(val)]["sample_id"] for i in range(30)],
                f"Latency sample mismatch: {k}")
        require((int(r["batch_size"]),int(r["image_size"]),int(r["warmup"]),r["device"])==(1,128,10,report["hardware"]),
                f"Timing protocol mismatch: {k}")
        vals=np.array([float(s["latency_ms"]) for s in seq])
        require((vals>0).all() and np.isfinite(vals).all(), f"Invalid latencies: {k}")
        for field,calc in (("mean_ms",np.mean(vals)),("median_ms",np.median(vals)),("p95_ms",np.quantile(vals,.95))):
            require(math.isclose(float(r[field]),float(calc),rel_tol=1e-9),f"Timing summary mismatch: {k}/{field}")
        output[k]=r
    target=out/"timing"
    target.mkdir(parents=True,exist_ok=True)
    for name in ("latency_raw.csv","latency_summary.csv","benchmark_report.json"):
        shutil.copyfile(src/name,target/name)
    return output


def test(inputs,out):
    a,b,c=folders(inputs)
    lock=obj(out/"results/protocol_lock.json")
    require(lock["status"]=="locked_before_D_test" and lock["protocol_sha256"]==digest(HERE/"protocol.md")
            and lock["source_releases"]==LOCK and lock["c_lock_sha256"]==digest(c/"protocol_lock.json"),"Invalid lock")
    timing=summarize_timing(c,a,out)
    diagnostics={(r["dataset"],r["category"],int(r["ratio_percent"])):r for r in csv_in(c/"per_category_results.csv")}
    per_sample=[]; per_cat=[]
    for setting in lock["settings"]:
        d,cat,ratio=setting["dataset"],setting["category"],setting["ratio_percent"]
        apath=a/"manifests"/(d+".json")
        require(digest(apath)==setting["a_manifest_sha256"],f"Manifest changed: {d}")
        truth=split(obj(apath),d,cat,"test")
        threshold=setting["threshold"]
        if ratio:
            folder=pc_folder(c,d,cat,ratio,"test")
            cfg=obj(folder/"model_config.json"); record=obj(folder/"test_record.json")
            require(cfg["bank_sha256"]==record["bank_sha256"]==setting["model_sha256"]
                    and cfg["manifest_sha256"]==setting["a_manifest_sha256"]
                    and record["predictions_sha256"]==digest(folder/"predictions.jsonl")
                    and record["protocol_sha256"]==lock["c_lock_sha256"]
                    and math.isclose(cfg["locked_threshold"],threshold,abs_tol=1e-8,rel_tol=1e-6),f"Test provenance: {folder}")
        else:
            folder=run_autoencoder(b,a,out,d,cat,truth,setting)
        paired=align(truth,read_predictions(folder))
        require(len({p["anomaly_map_path"] for _,p in paired})==len(paired),f"Duplicate map paths: {folder}")
        pixels=PixelAUROC(sum(r["width"]*r["height"] for r in truth))
        labels=[]; scores=[]; counts=Counter()
        for row,pred in paired:
            field=load_map(folder,row,pred)
            native=native_map(field,row["width"],row["height"])
            mask=native_mask(a,row)
            pixels.add(native,mask)
            score=float(pred["score"]); label=int(row["label"]); decision=int(score>threshold)
            counts[(label,decision)]+=1; labels.append(label); scores.append(score)
            root_prefix="C:project/" if ratio else "D:project/"
            map_ref=root_prefix+map_file(folder,pred).relative_to(c if ratio else out).as_posix()
            per_sample.append(dict(dataset=d,category=cat,model=setting["model"],sample_id=row["sample_id"],
                physical_id=row.get("physical_id",""),defect_type=row["defect_type"],label=label,score=score,
                threshold=threshold,predicted=decision,mask_fraction=float(mask.mean()),anomaly_map=map_ref))
        im_auc=image_auc(labels,scores); px_auc=pixels.compute()
        npos=pixels.positive_count; ntotal=len(pixels.values);del pixels
        if ratio:
            ref=diagnostics[(d,cat,ratio)]
            require(abs(im_auc-float(ref["image_auroc_diagnostic"]))<1e-10
                    and abs(px_auc-float(ref["pixel_auroc_diagnostic"]))<1e-7,f"C independent AUROC mismatch: {d}/{cat}/{ratio}: {im_auc}, {px_auc}, {ref}")
        tm=timing[(d,cat,ratio)]
        require(tm["artifact_sha256"]==setting["model_sha256"],f"Latency model differs: {d}/{cat}/{ratio}")
        item=dict(dataset=d,category=cat,model=setting["model"],ratio_percent=ratio,normal=len(labels)-sum(labels),
                  anomalous=sum(labels),positive_pixels=npos,negative_pixels=ntotal-npos,image_auroc=im_auc,
                  pixel_auroc=px_auc,threshold=threshold,tp=counts[(1,1)],fp=counts[(0,1)],tn=counts[(0,0)],fn=counts[(1,0)],
                  mean_ms=float(tm["mean_ms"]),median_ms=float(tm["median_ms"]),p95_ms=float(tm["p95_ms"]),
                  bank_patches=setting["bank_patches"],bank_bytes=setting["bank_bytes"],val_coverage_p95=setting["val_coverage_p95"])
        per_cat.append(item);print("CATEGORY",json.dumps(item),flush=True)
    macro=[]
    for d,cats in GROUPS.items():
        for ratio in RATIOS:
            group=[r for r in per_cat if r["dataset"]==d and r["ratio_percent"]==ratio]
            require({r["category"] for r in group}==set(cats) and len(group)==len(cats),f"Incomplete category coverage: {d}")
            item=dict(dataset=d,model=model_name(ratio),ratio_percent=ratio,categories=len(cats))
            for k in ("image_auroc","pixel_auroc","mean_ms"):
                item[k]=float(np.mean([r[k] for r in group]))
            for k in ("normal","anomalous","tp","fp","tn","fn","bank_bytes","bank_patches"):
                item[k]=sum(r[k] for r in group)
            macro.append(item);print("MACRO",json.dumps(item),flush=True)
    require(len(per_cat)==28 and len(per_sample)==2376, "Formal prediction coverage")
    csv_out(out/"results/per_category.csv",per_cat)
    csv_out(out/"results/per_dataset.csv",macro)
    csv_out(out/"results/per_sample.csv",per_sample)
    shutil.copyfile(b/"report/b_training_summary.csv",out/"results/b_training_summary.csv")
    json_out(out/"results/verification.json",dict(configurations=28,finite_test_predictions=len(per_sample),unique_test_ids=594,
        unique_validation_ids=256,thresholds=len(lock["settings"]),paired_timing_measurements=840,independently_matched_c_configurations=21,
        threshold_sha256=digest(out/"results/protocol_lock.json"),results_sha256=digest(out/"results/per_category.csv")))
    return per_cat,macro


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("lock","test"))
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    (freeze if args.stage=="lock" else test)(args.inputs.resolve(),args.out.resolve())
