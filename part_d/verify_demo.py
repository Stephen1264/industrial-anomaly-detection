"""Run the released Streamlit app and compare its two fixed demo examples with D's test export."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest
import torch

from prepare_demo import stage


def verify(b,output):
    sys.path.insert(0,str(b))
    from autoencoder.core import load_model,load_threshold,predict
    from prepare_inputs import digest
    test=output/"predictions/ae/mvtec_ad/bottle/test"
    ids={r["sample_id"]:r for r in map(json.loads,(test/"predictions.jsonl").read_text().splitlines())}
    with tempfile.TemporaryDirectory() as temporary:
        demo=Path(temporary)/"models";stage(b,output,demo)
        examples=json.loads((demo/"demo_samples/index.json").read_text(encoding="utf-8-sig"))
        weights=demo/"mvtec_ad/bottle/model.pt"
        model,config=load_model(weights)
        config["checkpoint_sha256"]=digest(weights)
        locked=load_threshold(weights.parent/"threshold.json",config)
        torch.set_num_threads(4)
        saved=sys.argv
        try:
            sys.argv=[str(b/"demo.py"),"--models",str(demo)]
            at=AppTest.from_file(str(b/"demo.py"),default_timeout=90).run()
            if at.exception:raise RuntimeError(str(at.exception))
            at.selectbox[0].select("mvtec_ad / bottle").run()
            results=[]
            for item in examples:
                pred=ids[item["sample_id"]]
                with Image.open(demo/"demo_samples"/item["file"]) as img:
                    formula,field,_=predict(model,config,img)
                cli_map=np.load(test/pred["anomaly_map_path"],allow_pickle=False)
                np.testing.assert_allclose(field,cli_map,rtol=1e-5,atol=2e-7)
                at.selectbox[1].select("缺陷示例" if item["label"] else "正常示例").run()
                if at.exception:raise RuntimeError(str(at.exception))
                web_score=float(at.metric[0].value)
                web_alarm=(formula>locked["threshold"])
                words=[m.value for m in at.markdown]
                if ("异常" if web_alarm else "正常") not in words:
                    raise ValueError(f"UI decision missing for {item['sample_id']}: {words}")
                if abs(web_score-pred["score"])>2e-6 or abs(formula-pred["score"])>2e-6 or web_alarm!=pred["is_anomaly"]:
                    raise ValueError(f"Demo/formal scoring differs for {item['sample_id']}")
                results.append(dict(sample_id=item["sample_id"],label=item["label"],web_score=web_score,
                    formal_score=pred["score"],locked_threshold=locked["threshold"],alarm=web_alarm,
                    maximum_map_difference=float(np.abs(field-cli_map).max())))
        finally:
            sys.argv=saved
    (output/"results/demo_verification.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    for r in results:print("DEMO",r)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b-project",type=Path,required=True);p.add_argument("--evaluation",type=Path,required=True)
    a=p.parse_args();verify(a.b_project.resolve(),a.evaluation.resolve())
