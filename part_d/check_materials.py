"""Validate document pages, speaker notes and computed headline results before packaging."""
import argparse
import base64
import csv
import io
import json
import math
import zipfile
from pathlib import Path

import pymupdf as fitz
from PIL import Image
from pptx import Presentation


def checks(out, slide_pdf):
    report=fitz.open(out/"report/Final_report.pdf")
    slides=fitz.open(slide_pdf)
    deck=Presentation(out/"presentation/Part_D_results.pptx")
    print("GENERATED PAGE COUNTS",len(report),len(slides),len(deck.slides),flush=True)
    assert len(slides)==len(deck.slides)==5, (len(slides),len(deck.slides))
    assert all(slide.has_notes_slide and len(slide.notes_slide.notes_text_frame.text.strip())>100 for slide in deck.slides)
    assert 7<=len(report)<=10
    for filename,pdf in (("report",report),("slides",slides)):
        thumbs=[]
        for idx,page in enumerate(pdf):
            spans=[s for block in page.get_text("dict")["blocks"] if "lines" in block
                   for line in block["lines"] for s in line["spans"] if s["text"].strip()]
            assert spans, (filename,idx,"empty")
            print("PAGE",filename,idx+1,len(page.get_text()),spans[0]["text"][:90])
            for s in spans:
                x0,y0,x1,y1=s["bbox"]
                assert x0>=-1 and y0>=-2 and x1<=page.rect.width+2 and y1<=page.rect.height+2,(filename,idx,s["text"],s["bbox"])
            pm=page.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
            im=Image.frombytes("RGB",(pm.width,pm.height),pm.samples)
            im.thumbnail((560,650) if filename=="report" else (650,420))
            thumbs.append(im)
        for start in range(0,len(thumbs),3 if filename=="report" else 2):
            group=thumbs[start:start+(3 if filename=="report" else 2)]
            contact=Image.new("RGB",(sum(k.width for k in group)+8*(len(group)-1),max(k.height for k in group)),"#cccccc")
            off=0
            for im in group:
                contact.paste(im,(off,0));off+=im.width+8
            f=io.BytesIO();contact.save(f,format="JPEG",quality=68,optimize=True)
            code=base64.b64encode(f.getvalue()).decode()
            for n in range(0,len(code),3500):
                print(f"PREVIEW {filename}-{start} {n:06d} {code[n:n+3500]}")
    whole=" ".join(p.get_text() for p in report)
    with (out/"results/per_dataset.csv").open(encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] in ("ae","pc05"):
                assert f"{float(r['image_auroc']):.4f}" in whole and f"{float(r['pixel_auroc']):.4f}" in whole
    assert len(json.loads((out/"results/demo_verification.json").read_text()))==2
    with zipfile.ZipFile(out/"report/Final_report.docx") as z:assert z.testzip() is None
    print(f"Materials verified: {len(report)} report pages and {len(deck.slides)} slides with notes.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("evaluation",type=Path);p.add_argument("slide_pdf",type=Path)
    args=p.parse_args();checks(args.evaluation,args.slide_pdf)
