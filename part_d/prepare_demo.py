"""Assemble B's unchanged demo with D's locked thresholds in a new directory."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def stage(b_project, evaluation, destination):
    if destination.exists():
        raise FileExistsError(f"Choose a new demo-model directory: {destination}")
    locked = sorted((evaluation/"demo/artifacts").glob("*/*/threshold.json"))
    if len(locked) != 7:
        raise ValueError("Expected seven D thresholds")
    weights = []
    for path in locked:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        relative = path.parent.relative_to(evaluation/"demo/artifacts")
        model = b_project/"artifacts"/relative/"model.pt"
        if cfg["status"] != "locked_by_D" or hashlib.sha256(model.read_bytes()).hexdigest() != cfg["checkpoint_sha256"]:
            raise ValueError(f"Wrong checkpoint or threshold: {relative}")
        weights.append((model,path,relative))
    samples = b_project/"artifacts/demo_samples"
    for filename in ("normal.png","anomaly.png","index.json"):
        if not (samples/filename).is_file():
            raise FileNotFoundError(samples/filename)
    destination.mkdir(parents=True)
    for model,path,relative in weights:
        target=destination/relative
        target.mkdir(parents=True)
        shutil.copyfile(model,target/"model.pt")
        shutil.copyfile(path,target/"threshold.json")
    shutil.copytree(samples,destination/"demo_samples")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b-project",type=Path,required=True)
    p.add_argument("--evaluation",type=Path,required=True)
    p.add_argument("--destination",type=Path,required=True)
    a=p.parse_args();stage(a.b_project,a.evaluation,a.destination)
