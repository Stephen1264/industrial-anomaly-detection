"""Build the five D presentation pages and their speaker notes from the evaluation."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from evaluate import folders,native_map,native_mask,obj
from report import rows,four,fmt


def case_detail(inputs,out):
    a,_,c=folders(inputs)
    records=rows(out/"results/per_sample.csv")
    choices=[("mvtec_ad","mvtec_ad/metal_nut/test/flip/017","Flipped metal nut"),
             ("local_capture","local_capture/nut/test_defect/nut_14_02","Controlled nut stain")]
    fig,ax=plt.subplots(2,4,figsize=(10.2,4.35))
    fig.subplots_adjust(left=.015,right=.99,top=.91,bottom=.07,wspace=.035,hspace=.27)
    for i,(dataset,sid,name) in enumerate(choices):
        row=next(r for r in obj(a/"manifests"/(dataset+".json"))["samples"] if r["sample_id"]==sid)
        mask=native_mask(a,row);h,w=mask.shape
        with Image.open(a/row["image_path"]) as im: rgb=np.asarray(im.convert("RGB"))
        bounds=(0,h,0,w)
        if i==1:
            yy,xx=np.where(mask)
            side=min(h,w,max(180,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*5)))
            x=int(np.clip((xx.min()+xx.max()-side)/2,0,w-side)); y=int(np.clip((yy.min()+yy.max()-side)/2,0,h-side))
            bounds=(y,y+side,x,x+side)
        y,z,x,u=bounds; crop=lambda field:field[y:z,x:u]
        panels=[crop(rgb),crop(mask)];titles=["Photograph","Defect mask"]
        for model,method in (("ae","Autoencoder"),("pc05","PatchCore 5%")):
            r=next(r for r in records if r["model"]==model and r["sample_id"]==sid)
            src,relative=r["anomaly_map"].split(":project/",1)
            field=native_map(np.load((out if src=="D" else c)/relative,allow_pickle=False),w,h)
            norm=crop(field)/max(float(np.max(field)),1e-12)
            heat=plt.get_cmap("inferno")(norm)[:,:,:3]
            panels.append(np.clip(.55*crop(rgb)/255+.45*heat,0,1))
            verdict="alarm" if r["predicted"]=="1" else "normal"
            titles.append(f"{method}   {verdict}")
        for j,(img,title) in enumerate(zip(panels,titles)):
            ax[i,j].imshow(img,cmap="gray",interpolation="nearest" if j==1 else "antialiased")
            ax[i,j].set_xticks([]);ax[i,j].set_yticks([])
            for s in ax[i,j].spines.values():s.set_visible(False)
            ax[i,j].set_title(title,fontsize=10,pad=3)
        ax[i,0].set_ylabel(name+("\nannotation-centered crop" if i else "\nfull view"),fontsize=9)
    fig.savefig(out/"figures/two_nut_cases.png",dpi=180,bbox_inches="tight",pad_inches=.12);plt.close(fig)


def native_deck(out, slides, metrics):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
    from pptx.util import Inches, Pt

    deck=Presentation()
    deck.slide_width=Inches(13+1/3);deck.slide_height=Inches(7.5)
    black=RGBColor.from_string("172E40");gray=RGBColor.from_string("42545F")
    teal=RGBColor.from_string("086D65");white=RGBColor(255,255,255)
    def box(slide,text,x,y,w,h,size=20,bold=False,color=black):
        shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
        frame=shape.text_frame;frame.word_wrap=True;frame.auto_size=MSO_AUTO_SIZE.NONE
        frame.margin_left=frame.margin_right=Inches(.01);frame.margin_top=frame.margin_bottom=Inches(.01)
        p=frame.paragraphs[0];p.text=text;p.font.name="Liberation Sans";p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color
        p.space_before=Pt(0);p.space_after=Pt(0);return shape
    def picture(slide,name,y,height):
        path=out/"figures"/name
        with Image.open(path) as im:ratio=im.width/im.height
        w=min(11.65,height*ratio);h=w/ratio
        slide.shapes.add_picture(str(path),Inches((13+1/3-w)/2),Inches(y+(height-h)/2),Inches(w),Inches(h))
    def table(slide,values,widths,y,row_height=.58):
        width=sum(widths);x=(13+1/3-width)/2
        shape=slide.shapes.add_table(len(values),len(widths),Inches(x),Inches(y),Inches(width),Inches(row_height*len(values)))
        tab=shape.table;tab.first_row=False;tab.horz_banding=False
        for j,w in enumerate(widths):tab.columns[j].width=Inches(w)
        for i,vals in enumerate(values):
            tab.rows[i].height=Inches(row_height)
            for j,v in enumerate(vals):
                cell=tab.cell(i,j);cell.margin_left=Inches(.13);cell.margin_right=Inches(.13)
                cell.margin_top=cell.margin_bottom=Inches(.045);cell.vertical_anchor=MSO_ANCHOR.MIDDLE
                cell.fill.solid();cell.fill.fore_color.rgb=teal if i==0 else RGBColor.from_string("EAF2F2" if i%2 else "F4F7F8")
                f=cell.text_frame;f.word_wrap=True;p=f.paragraphs[0];p.text=str(v);p.alignment=PP_ALIGN.LEFT if j==0 else PP_ALIGN.CENTER
                p.font.name="Liberation Sans";p.font.size=Pt(18);p.font.bold=i==0;p.font.color.rgb=white if i==0 else black
                p.space_before=p.space_after=Pt(0)
    for i,(title,content,speech) in enumerate(slides):
        slide=deck.slides.add_slide(deck.slide_layouts[6]);slide.background.fill.solid();slide.background.fill.fore_color.rgb=white
        box(slide,title,.62,.34,12.1,.66,31,True)
        if i==0:
            box(slide,"We paired each model prediction with the same test image and native defect mask.",.66,1.26,12,.65,21)
            table(slide,[("Test dataset","Normal","Defective","Test design"),("MVTec AD, five categories","152","392","Official test split"),
                         ("Local bottles and nuts","26","24","Held-out specimens")],(3.75,1.65,1.9,4.12),2.22,.70)
            box(slide,"Image and pixel AUROCs use continuous scores. The 99th percentile of each model’s normal validation scores sets the image alarm threshold.",.69,4.89,11.9,1.11,21)
        elif i==1:
            picture(slide,"category_auroc.png",1.17,4.80)
            box(slide,"MVTec image AUROC rose from 0.6658 to 0.9457. Wood favored the autoencoder for whole-image ranking.",.72,6.04,11.95,.77,18,False,gray)
        elif i==2:
            a=metrics["local_capture","ae"];p=metrics["local_capture","pc05"]
            table(slide,[("Local test result","Autoencoder","PatchCore 5%"),("Image AUROC",four(a["image_auroc"]),four(p["image_auroc"])),
                         ("Pixel AUROC",four(a["pixel_auroc"]),four(p["pixel_auroc"])),("Defects flagged","4 / 24","1 / 24"),("False alarms","5 / 26","0 / 26")],(4.55,3.42,3.42),1.48,.66)
            box(slide,"Each local threshold used 15 normal validation views of three objects. A high pixel ranking can coexist with few image-level alarms.",.78,5.35,11.7,.99,21)
        elif i==3:
            picture(slide,"coreset_auroc.png",1.20,4.23)
            box(slide,"MVTec latency was 16.69, 22.96 and 30.05 ms/image at 1%, 5% and 10%. The autoencoder took 2.33 ms/image on the same GPU.",.77,5.68,11.85,1.0,20)
        else:
            picture(slide,"two_nut_cases.png",1.18,4.96)
            box(slide,"The flipped nut covered 48.72% of its image. The local stain covered 0.047%. The lower row magnifies the annotated area for display.",.70,6.22,11.95,.73,17,False,gray)
        source=("Part A manifests and D protocol","D results/per_category.csv","D results/per_dataset.csv and thresholds.csv",
                "D coreset results and C’s RTX 4060 timing records","D results/case_index.csv. MVTec Software GmbH, CC BY-NC-SA 4.0. Local images by the project group.")[i]
        box(slide,source,.67,7.06,12,.26,10,False,gray)
        slide.notes_slide.notes_text_frame.text=speech
    deck.core_properties.title="Industrial anomaly detection and localization: evaluation results"
    deck.core_properties.author="BNBU Deep Learning course project group"
    deck.save(out/"presentation/Part_D_results.pptx")


def build(inputs,out):
    case_detail(inputs,out)
    directory=out/"presentation";directory.mkdir(exist_ok=True)
    ds={(r["dataset"],r["model"]):r for r in rows(out/"results/per_dataset.csv")}
    a,p=ds["local_capture","ae"],ds["local_capture","pc05"]
    slides=[
    ("How we compared 594 test images",
     """| Test dataset | Normal | Defective | Test design |
|---|---:|---:|---|
| MVTec AD, five categories | 152 | 392 | Official test image |
| Local bottles and nuts | 26 | 24 | Held-out specimens |

We froze one threshold per category and configuration on normal validation images. We ranked images and native-resolution pixels using continuous scores.""",
     "We compared both models on the same image IDs. The public benchmark kept its original test split. For the local set, every photograph of one object stayed in one partition. We scored each image and upsampled each continuous map to the native mask size. We then averaged category AUROCs within each dataset. The thresholds were fixed before D evaluated the tests."),
    ("PatchCore led in four of five MVTec categories",
     "![Image and pixel AUROC by category](../figures/category_auroc.png){height=2.85in}\n\nMVTec mean image AUROC rose from 0.6658 to 0.9457. Wood favored the autoencoder for image ranking.",
     "The orange circles represent the autoencoder, and teal diamonds show the 5% PatchCore bank. The largest image-level gain was on metal nuts, from 0.4399 to 0.9712. Wood was the exception: the autoencoder ranked whole images better, while PatchCore had the higher pixel AUROC. The ranking task and the localization task can favor different methods."),
    ("Only one of 24 stains raised a PatchCore alarm",
     "| Local test result | Autoencoder | PatchCore 5% |\n|---|---:|---:|\n"
     +f"| Image AUROC | {four(a['image_auroc'])} | {four(p['image_auroc'])} |\n"
     +f"| Pixel AUROC | {four(a['pixel_auroc'])} | {four(p['pixel_auroc'])} |\n"
     +"| Defects flagged | 4 / 24 | 1 / 24 |\n| False alarms | 5 / 26 | 0 / 26 |\n\nEach local threshold used 15 normal validation views of three physical objects.",
     "PatchCore assigned higher scores to many marked pixels than to background pixels, which gave it a high localization AUROC. To raise an alarm, however, its image score had to exceed the 99th percentile of normal validation scores. Only one stained photograph cleared that threshold. The local data contain repeated views, so this is evidence from five test objects per category."),
    ("A larger feature bank gave mixed returns",
     "![Image and pixel AUROC by retained feature fraction](../figures/coreset_auroc.png){height=2.55in}\n\nOn MVTec AD, 1%, 5% and 10% banks took 16.69, 22.96 and 30.05 ms per image. The autoencoder took 2.33 ms.",
     "On MVTec AD, retaining 1% of normal patches gave the highest image AUROC, while 5% gave the highest pixel AUROC. Doubling the bank again gave little localization change and increased the timed cost. Each configuration used the same RTX 4060, a batch size of one, ten warm-ups and 30 synchronized measurements. The timer excluded image decoding, loading and resizing."),
    ("Two nut defects drew different decisions",
     "![Large flipped nut and a small controlled stain](../figures/two_nut_cases.png){height=3.05in}\n\nThe flipped nut covered 48.72% of the image. The local stain covered 0.047%. The crop is for display only.",
     "The top row shows a flipped metal nut. PatchCore flagged it and the autoencoder missed it. The bottom row magnifies the annotated stain on a held-out local nut. Both image scores fell below their thresholds. All metrics use the full images. For these products, the final design choice depends on both the cost of missing a defect at the chosen threshold and the time available for inspection.")
    ]
    native_deck(out,slides,ds)
    body=[];notes=["# Part D speaker notes",""]
    for no,(title,content,speech) in enumerate(slides,1):
        body.extend(["## "+title,"",content,"","::: notes",speech,":::",""])
        notes.extend([f"## {no}. {title}","",speech,""])
    (directory/"slides.md").write_text("\n".join(body),encoding="utf-8")
    (directory/"speaker_notes.md").write_text("\n".join(notes),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();build(args.inputs.resolve(),args.out.resolve())
