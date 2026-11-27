"""Create the final course report and figures from D's evaluated outputs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor

SETS={"mvtec_ad":"MVTec AD", "local_capture":"Local"}
NAMES={"bottle":"Bottle","cable":"Cable","metal_nut":"Metal nut","tile":"Tile","wood":"Wood","nut":"Nut"}
DARK="#243E56"
TEAL="#00796D"
OCHRE="#B65D26"


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def pct(x):
    return f"{float(x)*100:.2f}%"


def four(x):
    return f"{float(x):.4f}"


def fmt(x):
    return f"{float(x):.2f}"


def figures(out,categories,macro):
    directory=out/"figures";directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "font.family":"DejaVu Sans","savefig.facecolor":"white","axes.titleweight":"bold"})
    mains=[r for r in categories if r["model"]=="ae"]
    lookup={(r["dataset"],r["category"],r["model"]):r for r in categories}
    labels=[NAMES[r["category"]]+(" (local)" if r["dataset"]=="local_capture" else "") for r in mains]
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.45),sharey=True)
    fig.subplots_adjust(left=.19,right=.975,top=.85,bottom=.22,wspace=.08)
    for col,(ax,metric,title) in enumerate(zip(axes,("image_auroc","pixel_auroc"),("Image AUROC","Pixel AUROC"))):
        for i,ae in enumerate(mains):
            pc=lookup[ae["dataset"],ae["category"],"pc05"]
            x,y=float(ae[metric]),float(pc[metric]); ax.plot([x,y],[i,i],color="#B6C2CA",linewidth=2,zorder=1)
            ax.scatter(x,i,color=OCHRE,s=43,zorder=3,label="Autoencoder" if i==0 else None)
            ax.scatter(y,i,color=TEAL,s=48,zorder=3,marker="D",label="PatchCore 5%" if i==0 else None)
        ax.axvline(.5,color="#A8ACB2",ls=":",lw=1)
        ax.axhline(4.5,color="#C4CDD1",lw=.8)
        ax.set(xlim=(.32,1.025),xticks=[.4,.5,.6,.7,.8,.9,1],yticks=range(len(mains)),title=title)
        ax.set_yticklabels(labels);ax.grid(axis="x",alpha=.18);ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
    axes[0].invert_yaxis()
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,.01))
    fig.savefig(directory/"category_auroc.png",dpi=175);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8))
    fig.subplots_adjust(left=.08,right=.985,top=.87,bottom=.24,wspace=.25)
    encoding=[("image_auroc","Image AUROC",TEAL,"o"),("pixel_auroc","Pixel AUROC",DARK,"s")]
    for ax,(d,name) in zip(axes,SETS.items()):
        data=[r for r in macro if r["dataset"]==d and r["model"]!="ae"]
        x=[int(r["ratio_percent"]) for r in data]
        for metric,label,color,mark in encoding:
            ax.plot(x,[float(r[metric]) for r in data],color=color,marker=mark,label=label,lw=2,ms=6)
        ax.set(xlabel="Retained normal patches (%)",title=name,xticks=x,ylim=(.62,1.01),yticks=[.7,.8,.9,1])
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Category macro AUROC")
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,-.015))
    fig.savefig(directory/"coreset_auroc.png",dpi=175);plt.close(fig)


def link(paragraph, label, url):
    h=OxmlElement("w:hyperlink")
    h.set(qn("r:id"),paragraph.part.relate_to(url,RT.HYPERLINK,is_external=True))
    run=OxmlElement("w:r");pr=OxmlElement("w:rPr");color=OxmlElement("w:color");color.set(qn("w:val"),"176F76");pr.append(color)
    run.append(pr);t=OxmlElement("w:t");t.text=label;run.append(t);h.append(run);paragraph._p.append(h)


class Paper:
    def __init__(self,folder):
        self.folder=folder
        self.doc=Document();self.md=[]
        sec=self.doc.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
        sec.top_margin=Inches(.70);sec.bottom_margin=Inches(.69);sec.left_margin=sec.right_margin=Inches(.83)
        sec.footer_distance=Inches(.28)
        for key,size in (("Normal",10.5),("Title",25),("Heading 1",19),("Heading 2",12),("Caption",9)):
            s=self.doc.styles[key];s.font.name="Liberation Sans";s.font.size=Pt(size);s.font.color.rgb=RGBColor(24,32,39) if key in ("Normal","Caption") else RGBColor(0,0,0)
            pf=s.paragraph_format;pf.space_before=Pt(0 if key in ("Normal","Title","Heading 1") else 6)
            pf.space_after=Pt(5 if key in ("Normal","Heading 2","Caption") else 7)
            pf.line_spacing=1.12 if key=="Normal" else 1.04
            if key in ("Title","Heading 1","Heading 2"):
                s.font.bold=True
                ppr=s.element.get_or_add_pPr()
                for border in list(ppr.findall(qn("w:pBdr"))):ppr.remove(border)
            if key=="Caption":s.font.italic=False
        foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.RIGHT
        run=foot.add_run("BNBU  ·  Deep Learning  ·  Fall 2026     ");run.font.size=Pt(8)
        field=OxmlElement("w:fldSimple");field.set(qn("w:instr"),"PAGE");foot._p.append(field)
        self.doc.core_properties.title="Industrial Anomaly Detection and Localization"
        self.doc.core_properties.author="BNBU Deep Learning course project group"

    def heading(self,text,level=2):
        self.doc.add_heading(text,level=level);self.md.extend(["#"*(level+1)+" "+text,""])

    def page(self,text):
        p=self.doc.add_heading(text,level=1);p.paragraph_format.page_break_before=True
        self.md.extend(["## "+text,""])

    def paragraph(self,txt):
        self.doc.add_paragraph(txt);self.md.extend([txt,""])

    def title(self,t):
        self.doc.add_paragraph(t,style="Title");self.md.extend(["# "+t,""])

    def table(self,headers,data,widths):
        t=self.doc.add_table(rows=0,cols=len(headers));t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
        for c,w in zip(t.columns,widths):c.width=Inches(w)
        pr=t._tbl.tblPr
        borders=OxmlElement("w:tblBorders")
        for side in ("top","bottom","left","right","insideH","insideV"):
            br=OxmlElement("w:"+side);br.set(qn("w:val"),"single");br.set(qn("w:sz"),"4");br.set(qn("w:color"),"D9D9D9");borders.append(br)
        pr.append(borders)
        space=OxmlElement("w:tblCellMar")
        for side,n in (("top",38),("bottom",38),("left",72),("right",72)):
            el=OxmlElement("w:"+side);el.set(qn("w:w"),str(n));el.set(qn("w:type"),"dxa");space.append(el)
        pr.append(space)
        for i,values in enumerate([headers]+list(data)):
            row=t.add_row()
            if i==0:row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
            for j,(cell,val,w) in enumerate(zip(row.cells,values,widths)):
                cell.width=Inches(w);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                p=cell.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.04
                run=p.add_run(str(val));run.font.size=Pt(9.2);run.font.bold=i==0
                shade=OxmlElement("w:shd");shade.set(qn("w:fill"),"E4ECEF" if i==0 else ("F6F8F9" if i%2==0 else "FFFFFF"));cell._tc.get_or_add_tcPr().append(shade)
        self.md+=["| "+" | ".join(map(str,headers))+" |","| "+" | ".join(["---"]*len(headers))+" |"]
        self.md+=["| "+" | ".join(map(str,r))+" |" for r in data]+[""]

    def fig(self,path,caption,width):
        p=self.doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after=Pt(1);p.paragraph_format.keep_with_next=True
        p.add_run().add_picture(str(path),width=Inches(width))
        self.doc.add_paragraph(caption,style="Caption")
        self.md+=["!["+caption+"]("+("../figures/"+path.name)+")",""]

    def ref(self,label,url):
        p=self.doc.add_paragraph();link(p,label,url);self.md+=["["+label+"]("+url+")",""]

    def save(self):
        self.doc.save(self.folder/"Final_report.docx")
        (self.folder/"Final_report.md").write_text("\n".join(self.md),encoding="utf-8")


def build(out):
    folder=out/"report";folder.mkdir(exist_ok=True)
    categories=rows(out/"results/per_category.csv");macro=rows(out/"results/per_dataset.csv")
    figures(out,categories,macro)
    paper=Paper(folder)
    paper.title("Industrial Anomaly Detection and Localization")
    paper.paragraph("BNBU Deep Learning  |  Fall 2026 course project")
    paper.heading("Abstract")
    paper.paragraph("We trained a convolutional autoencoder and built a PatchCore feature bank separately for each product category, using only normal training images. We evaluated 544 official MVTec AD test images and 50 independently held-out group photographs. On the five MVTec categories, the 5% PatchCore model reached 0.9457 image AUROC and 0.9440 pixel AUROC, compared with 0.6658 and 0.6691 for the autoencoder. Its average inference time increased from 2.33 to 22.96 ms per image on the same GPU. On the local set, PatchCore achieved 0.9526 pixel AUROC but raised an alarm on only one of 24 stained photographs at the prespecified threshold.")
    paper.heading("Motivation and data")
    paper.paragraph("An inspection system must decide which items to flag and tell an inspector where to look. Normal products are easier to collect than representative examples of every defect. We therefore learned each category’s normal appearance and evaluated two questions separately: can the model rank a defective image above a normal image, and can its map rank defective pixels above background?")
    paper.table(["Source / category","Normal train","Normal val.","Normal test","Defect test"],[
        ("MVTec bottle",167,42,20,63),("MVTec cable",179,45,58,92),("MVTec metal nut",176,44,22,93),
        ("MVTec tile",184,46,33,84),("MVTec wood",198,49,19,60),("Local bottle",40,15,13,12),
        ("Local nut",40,15,13,12)],(2.10,1.18,1.12,1.18,1.15))
    paper.paragraph("MVTec AD supplies normal training images, test images and native defect masks [1]. We kept the official tests and used a deterministic per-category holdout of about 20% of normal training images (seed 42). The group photographed 16 physical bottles and 16 nuts from five angles each. Within each local category, eight objects were assigned to training, three to validation and five to testing. The 24 controlled-stain masks were traced at native resolution. Six before-stain photographs of test objects retain normal labels.")

    paper.page("Models and evaluation protocol")
    paper.heading("Reconstruction baseline")
    paper.paragraph("Reconstruction methods fit normal appearance and use residual error as evidence of an anomaly [2]. Our autoencoder mapped 128 × 128 RGB inputs to a 64 × 8 × 8 latent representation through four stride-two convolutions. A mirrored transposed-convolution decoder reconstructed the input. Hidden layers used ReLU and the output used sigmoid. Each category used 30 epochs, Adam with learning rate 0.001, batch size 16 and seed 42. We selected the checkpoint with the lowest normal-validation mean squared error. The pixel score was the channel-mean squared residual. The image score averaged the highest 1% of pixel scores.")
    paper.heading("PatchCore feature model")
    paper.paragraph("PatchCore represents normal products with a memory bank of local pretrained features [3]. Our implementation froze an ImageNet-pretrained Wide ResNet-50-2. It pooled features from layers 2 and 3 over 3 × 3 neighborhoods, aligned them on a 16 × 16 patch grid and concatenated them into 1,536-dimensional vectors. A farthest-first selection on a 128-dimensional Gaussian projection retained 1%, 5% or 10% of the normal patches. At test time, each patch used its exact nearest-neighbor Euclidean distance in the original feature space. The maximum raw distance scored the image. Bilinear upsampling and Gaussian smoothing (sigma 4) produced the 128 × 128 map. We used C’s designated 5% configuration for the main comparison.")
    paper.heading("Scoring and provenance")
    paper.paragraph("Part D froze the 99th percentile of normal-validation image scores independently for each dataset, category and model. A score strictly above this value raised an alarm. We checked all 28 validation configurations before exporting B’s 594 test predictions. C had already published its test diagnostics. We retained its existing configuration and independently evaluated all 1,782 C test predictions. A and B’s differently serialized manifests had identical parsed content. Checkpoint and feature-bank hashes link the models to their predictions.")
    paper.paragraph("We joined predictions by sample ID. Each continuous map was bilinearly interpolated to the original image dimensions. Normal images contributed all-zero masks. Image AUROC ranks positive versus normal images. Pixel AUROC pools foreground and background pixels within a category, assigning half credit for ties. The reported macro mean weights each category equally within its dataset. There is no pixel threshold.")
    paper.heading("Timing")
    paper.paragraph("We recalculated summary statistics from C’s 840 synchronized GPU timings. Each configuration used the same 128 × 128 inputs, batch size one, ten warm-ups and 30 timed calls cycling through up to 15 normal validation images. Timing on an RTX 4060 Laptop GPU included model normalization, CPU/GPU transfers, scoring and map generation. Loading weights, image decoding, resizing and visualization were outside the timed interval.")

    paper.page("Test performance by product category")
    paper.paragraph("The 5% PatchCore configuration improved image AUROC in four of five MVTec categories and pixel AUROC in all five. Metal nuts showed the largest image-level gain, from 0.4399 to 0.9712. On wood, the autoencoder ranked images better (0.9596 versus 0.9175), while PatchCore localized defects better (0.8855 versus 0.7331).")
    index={(r["dataset"],r["category"],r["model"]):r for r in categories}
    tab=[]
    for r in categories:
        if r["model"]!="ae":continue
        q=index[r["dataset"],r["category"],"pc05"]
        tab.append((("Local " if r["dataset"]=="local_capture" else "")+NAMES[r["category"]],
                    four(r["image_auroc"]),four(q["image_auroc"]),four(r["pixel_auroc"]),four(q["pixel_auroc"])))
    paper.table(["Category","AE image","PC5 image","AE pixel","PC5 pixel"],tab,(1.68,1.25,1.25,1.25,1.25))
    paper.fig(out/"figures/category_auroc.png","Figure 1. Paired AUROCs. The dotted line marks chance ranking. The two local categories use independently fitted models.",6.50)
    paper.paragraph("Across the five MVTec categories, PatchCore gained 0.2800 in image AUROC and 0.2750 in pixel AUROC. The largest pixel-level gain was on tile (0.4475 to 0.9323). This is compatible with the reconstruction model responding to normal texture variation. These experiments compare complete pipelines, including the feature model’s ImageNet pretraining.")

    paper.page("Alarms and inference cost")
    paper.paragraph("On MVTec AD, the 5% model flagged 315 of 392 defective images and 10 of 152 normal images. The autoencoder flagged 67 defects and three normal images. On the local set, the same validation policy produced only one PatchCore detection and no false alarms. Local pixel AUROC remained high because it measures the ordering of pixels across each category, while an image alarm depends on a single score clearing a separate cutoff.")
    primary=[r for r in macro if r["model"] in ("ae","pc05")]
    paper.table(["Set / method","Image AUC","Pixel AUC","TP / defects","FP / normal","ms/image"],
                [(SETS[r["dataset"]]+" / "+("AE" if r["model"]=="ae" else "PC5"),four(r["image_auroc"]),four(r["pixel_auroc"]),
                  f'{r["tp"]} / {r["anomalous"]}',f'{r["fp"]} / {r["normal"]}',fmt(r["mean_ms"])) for r in primary],
                (1.46,.98,.99,1.14,1.12,1.0))
    paper.heading("Feature-bank size")
    paper.paragraph("On the normal MVTec validation images, the category mean of the 95th-percentile patch-to-bank distance fell from 1.916 at 1% to 1.658 at 5% and 1.584 at 10%. The smaller second improvement illustrates the coverage trade-off behind the prespecified 5% bank. Test image AUROC was highest at 1%, while localization peaked at 5%. Moving from 5% to 10% roughly doubled the stored vectors and increased mean latency from 22.96 to 30.05 ms with little change in localization. On the local set, 1% ranked images best and 5% ranked pixels best. The 10% timing was slightly below the 5% timing locally. A single benchmark does not establish a monotonic latency trend.")
    paper.table(["Set / bank","Image AUC","Pixel AUC","Bank MiB","ms/image"],
                [(SETS[r["dataset"]]+" / "+r["ratio_percent"]+"%",four(r["image_auroc"]),four(r["pixel_auroc"]),
                  fmt(float(r["bank_bytes"])/2**20),fmt(r["mean_ms"])) for r in macro if r["model"]!="ae"],
                (1.65,1.22,1.25,1.22,1.35))
    paper.paragraph("Bank size sums the feature-vector bytes for the categories in each dataset; it excludes the CNN and working memory. Relative to the autoencoder, 5% PatchCore took 9.85 times as long on MVTec AD and 7.18 times as long locally. The image-classification and pixel-localization objectives therefore give different reasons to choose a bank size.")

    details={
      "mvtec_ad":("MVTec defect and false-alarm cases","The flipped metal nut occupied 48.72% of the image. The autoencoder missed it, while PatchCore raised an alarm. Both missed the cable insulation puncture, whose mask occupied 0.144% of the image. PatchCore also flagged the normal metal nut shown in the last row. These are selected illustrations, not frequency estimates."),
      "local_capture":("Controlled stains on held-out objects","Even the largest local stain occupied only 0.247% of its image. PatchCore flagged that bottle at 3.5045 against a threshold of 3.4226. Both models missed the small nut stain (0.047% of the image). The normal bottle in the bottom row triggered the autoencoder at 0.1937 against 0.1411.")}
    for d,(head,interpretation) in details.items():
        paper.page(head)
        paper.paragraph(interpretation)
        paper.fig(out/"figures"/(d+"_case_panel.png"),"Figure "+("2" if d=="mvtec_ad" else "3")+
                  ". Each row shows the original image, native annotation and both model maps. Colors are scaled within each panel for visibility. Scores, thresholds and full sample IDs are recorded in results/case_index.csv.",5.82)

    paper.page("Conclusions and scope")
    paper.paragraph("For the five benchmark categories, 5% PatchCore offered the stronger overall ranking and localization at higher inference cost. The autoencoder remained competitive for detecting defects in wood and was faster in every category. For the local photographs, a high pixel AUROC coexisted with poor image-level recall. The intended use therefore determines which finding matters: a position map can guide inspection, but an automated reject decision needs an independently validated operating threshold.")
    paper.heading("Limits of the evidence")
    paper.paragraph("The local experiment contained only five independent test specimens per category. Multiple views of one specimen are correlated, and controlled stains cover a narrow range of defects. Each local threshold was estimated from only 15 normal photographs of three objects. Diffuse stain boundaries were traced approximately. The 128 × 128 model input can remove small localized details before the map is enlarged to native size.")
    paper.paragraph("The results cover one seed, five public categories, and one timing setup. A wider study could hold out additional physical objects and acquisition conditions, then select an operating point on a separate development set with representative defects. The present test sets remain reserved for this comparison.")
    paper.heading("Demo and reproducibility")
    paper.paragraph("B’s Streamlit demo uses the seven released checkpoints with D’s category-specific thresholds. Its fixed MVTec bottle example with a defect is a useful failure case: its reconstruction score is approximately 0.0530, below the 0.0757 bottle threshold. The paired UI and script checks are recorded in results/demo_verification.json. The Chinese handoff supplies commands to reproduce the evaluation and launch the demo.")
    paper.paragraph("The three input releases are pinned by SHA-256. results/protocol_lock.json records model fingerprints, validation inputs and numeric thresholds. results/per_sample.csv identifies the source of every map. The raw timing measurements, all seven autoencoder test exports and the 21 independent PatchCore checks accompany the report.")
    paper.heading("References and data credit")
    paper.ref("[1] Bergmann et al. MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. CVPR, 2019.","https://doi.org/10.1109/CVPR.2019.00982")
    paper.ref("[2] Baur et al. Autoencoders for unsupervised anomaly segmentation in brain MR images: A comparative study. Medical Image Analysis 69, 2021.","https://doi.org/10.1016/j.media.2020.101952")
    paper.ref("[3] Roth et al. Towards Total Recall in Industrial Anomaly Detection. CVPR, 2022.","https://doi.org/10.1109/CVPR52688.2022.01392")
    paper.ref("MVTec Software GmbH. MVTec AD images and derived examples, CC BY-NC-SA 4.0. Local photographs and masks are credited to the course project group.","https://www.mvtec.com/research-teaching/datasets/mvtec-ad")
    paper.save()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("evaluation",type=Path)
    build(parser.parse_args().evaluation.resolve())
