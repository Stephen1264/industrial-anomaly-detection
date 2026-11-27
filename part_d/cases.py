"""Make paired case panels after test metrics, with ID, decision and mask area recorded."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from evaluate import GROUPS, csv_in, csv_out, folders, obj
from metrics import native_mask, native_map


def select(pairs):
    normals=[p for p in pairs if p[0]["label"]=="0"]
    anomalies=[p for p in pairs if p[0]["label"]=="1"]
    tn=[p for p in normals if all(r["predicted"]=="0" for r in p)]
    false_alarms=[p for p in normals if any(r["predicted"]=="1" for r in p)]
    def margin(pair):
        return max(float(r["score"])/max(float(r["threshold"]),1e-12) for r in pair)
    typical=sorted(tn or normals,key=lambda p:(margin(p),p[0]["sample_id"]))
    key=lambda p:(float(p[0]["mask_fraction"]),p[0]["sample_id"])
    cases=[("normal",typical[len(typical)//2]),("large defect",max(anomalies,key=key)),
           ("small defect",min(anomalies,key=key))]
    if false_alarms:
        cases.append(("false positive",max(false_alarms,key=lambda p:(margin(p),p[0]["sample_id"]))))
    return cases


def build(inputs,out):
    a,_,c=folders(inputs)
    entries=csv_in(out/"results/per_sample.csv")
    dest=out/"figures";dest.mkdir(parents=True,exist_ok=True)
    index=[]
    for d in GROUPS:
        rows={r["sample_id"]:r for r in obj(a/"manifests"/(d+".json"))["samples"] if r["split"]=="test"}
        models={m:{r["sample_id"]:r for r in entries if r["dataset"]==d and r["model"]==m} for m in ("ae","pc05")}
        pairs=[(models["ae"][k],models["pc05"][k]) for k in sorted(rows)]
        selected=select(pairs)
        fig,axes=plt.subplots(len(selected),4,figsize=(12,3.3*len(selected)),squeeze=False)
        fig.subplots_adjust(top=.951,bottom=.048,hspace=.38,wspace=.04)
        for n,(kind,pair) in enumerate(selected):
            sid=pair[0]["sample_id"]; row=rows[sid]; w,h=row["width"],row["height"]
            with Image.open(a/row["image_path"]) as im:
                im=im.convert("RGB"); im.thumbnail((560,560),Image.Resampling.LANCZOS); rgb=np.array(im)
            display=(rgb.shape[1],rgb.shape[0])
            mask=np.asarray(Image.fromarray(native_mask(a,row)).resize(display,Image.Resampling.NEAREST))
            panels=[rgb,np.repeat((mask.astype("uint8")*255)[:,:,None],3,axis=2)]
            captions=["Photograph","Ground-truth mask"]
            for item,name in zip(pair,("Autoencoder","PatchCore 5%")):
                source,relative=item["anomaly_map"].split(":project/",1)
                path=(out if source=="D" else c)/relative
                field=native_map(np.load(path,allow_pickle=False),*display)
                color=plt.get_cmap("inferno")(field/max(float(field.max()),1e-12))[:,:,:3]
                panels.append(np.clip(.56*rgb/255+.44*color,0,1))
                captions.append(f"{name}   {'alarm' if item['predicted']=='1' else 'normal'}\n"
                                f"score {float(item['score']):.4f}   threshold {float(item['threshold']):.4f}")
            for j,(panel,caption) in enumerate(zip(panels,captions)):
                ax=axes[n,j];ax.imshow(panel);ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                ax.set_title(caption,fontsize=9)
            axes[n,0].set_xlabel(f"{kind.upper()}  |  {row['category']}  |  mask {float(pair[0]['mask_fraction'])*100:.3f}%\n{sid}",fontsize=8)
            index.append(dict(dataset=d,case=kind,sample_id=sid,category=row["category"],label=row["label"],
                              defect_type=row["defect_type"],mask_fraction=pair[0]["mask_fraction"],
                              ae_score=pair[0]["score"],ae_threshold=pair[0]["threshold"],ae_alarm=pair[0]["predicted"],
                              pc05_score=pair[1]["score"],pc05_threshold=pair[1]["threshold"],pc05_alarm=pair[1]["predicted"],
                              figure=f"figures/{d}_case_panel.png"))
        dsname="MVTec AD" if d=="mvtec_ad" else "Group-collected bottle and nut set"
        fig.suptitle(dsname+" | Paired detection and localization cases",fontsize=15,fontweight="bold")
        fig.text(.04,.012,"Heatmaps are scaled to each panel's own maximum for display. AUROC uses raw scores.  "
                 +("MVTec Software GmbH, CC BY-NC-SA 4.0." if d=="mvtec_ad" else "Photographs and masks: course project group."),fontsize=8)
        fig.savefig(dest/(d+"_case_panel.png"),dpi=145,bbox_inches="tight");plt.close(fig)
    csv_out(out/"results/case_index.csv",index)
    for row in index: print("CASE",row)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();build(args.inputs.resolve(),args.out.resolve())
