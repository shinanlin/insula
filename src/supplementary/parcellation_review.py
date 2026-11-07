"""Native MRI review pages for frozen sites with regenerated-label conflicts."""
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
import base64
import hashlib
import json
import textwrap
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/"results/audit/figure5_anatomy_review_2026-10-20"
RESULTS=ROOT/"results/supplementary/parcellation_conflict_review"
IMAGES=ROOT/"img/supplementary/parcellation_conflict_review"
FLAG="#D55E00"
CONTACT="#2369BD"
MIDPOINT="#6A5ACD"
TISSUE={"白质":"White matter","灰质":"Gray matter","CSF":"CSF","背景/未分类":"Unassigned/background"}
EVIDENCE={
 "native_unknown_retained_user_policy":"Native Unknown; retained by reviewed rule",
 "native_aparc_aseg_ribbon_WM":"Native aparc/aseg/ribbon: white matter",
 "consistent_archived_Hammers_insula_GM":"Archived Hammers: insula gray matter"
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def safe_rows():
    source=AUDIT/"29_sites_by_subject_task_old_new_labels.csv"
    table=pd.read_csv(source,keep_default_na=False)
    assert len(table)==50 and table.site_id.nunique()==29
    subjects=json.loads((ROOT/"results/maper_restore_2026-10-18/batch_manifest.json").read_text())["subjects"]
    subjects={r["subject"]:r for r in subjects}
    result=[]
    for site,group in table.groupby("site_id",sort=True):
        first=group.iloc[0].to_dict()
        assert re.fullmatch(r"D\d{4}_[A-Za-z0-9]+\d+-\d+",site),site
        consistent=["subject","contact_1","contact_2","old_midpoint_label","new_midpoint_label",
                    "old_contact_1_label","new_contact_1_label","old_contact_2_label","new_contact_2_label",
                    "new_contact_1_tissue","new_contact_2_tissue","conflict_reason","new_CSV_sha256"]
        # A different task's CSV may have different unrelated rows; compare the
        # selected coordinate and label fields instead of requiring CSV identity.
        consistent.remove("new_CSV_sha256")
        for field in consistent:
            assert group[field].nunique()==1,(site,field)
        entries=[]
        for row in group.to_dict("records"):
            path=Path(row["new_CSV"])
            assert sha(path)==row["new_CSV_sha256"],path
            meta=pd.read_csv(path,keep_default_na=False).set_index("name").loc[site]
            entries.append(meta)
        positions=np.array([[entries[0][a+"1"] for a in "xyz"],
                            [entries[0][a] for a in "xyz"],
                            [entries[0][a+"2"] for a in "xyz"]],float)
        for meta in entries[1:]:
            other=np.array([[meta[a+"1"] for a in "xyz"],[meta[a] for a in "xyz"],[meta[a+"2"] for a in "xyz"]],float)
            np.testing.assert_allclose(other,positions,atol=1e-9,rtol=0)
        np.testing.assert_allclose(positions[1],(positions[0]+positions[2])/2,atol=1e-9,rtol=0)
        sub=subjects[first["subject"]]
        first.update(tasks=sorted(group.task.unique()),task_ids=sorted(group.task_id.unique()),
            task_records=group[["task","native_input_state"]].to_dict("records"),
            positions=positions.tolist(),MRI=sub["orig_path"],reconstruction=sub["reconstruction"])
        result.append(first)
    return result

@lru_cache(maxsize=3)
def load_mri(path):
    image=nib.load(path)
    volume=np.asarray(image.dataobj,dtype=np.float32)
    assert volume.ndim==3 and np.isfinite(volume).all()
    affine=image.header.get_vox2ras_tkr()
    inverse=np.linalg.inv(affine)
    positive=volume[volume>0]
    limits=np.percentile(positive,[1,99])
    return {"volume":volume,"tkr_to_voxel":inverse,"voxel_to_tkr":affine,
            "limits":limits,"path":str(path),"shape":image.shape}

def sample_native(mri,point,u,v,half,spacing,global_context=False):
    point=np.asarray(point,float)
    u=np.asarray(u,float);v=np.asarray(v,float)
    if global_context:
        center=point-np.dot(point,u)*u-np.dot(point,v)*v
    else:
        center=point
    n=int(round(2*half/spacing))+1
    grid=np.linspace(-half,half,n)
    xx,yy=np.meshgrid(grid,grid,indexing="xy")
    world=center+xx[...,None]*u+yy[...,None]*v
    vox=nib.affines.apply_affine(mri["tkr_to_voxel"],world)
    data=map_coordinates(mri["volume"],vox.reshape(-1,3).T,order=1,
                         mode="constant",cval=0).reshape(xx.shape)
    return grid,data,center

def marker(ax,x,y,point):
    ax.scatter([x],[y],s=9 if point["role"]=="midpoint" else 7,
       marker="D" if point["role"]=="midpoint" else "o",
       facecolors="none",edgecolors=point["color"],linewidths=.65,zorder=5)
    for sign in [-1,1]:
        ax.plot([x+sign*1.5,x+sign*4],[y,y],color=point["color"],lw=.45,zorder=4)
        ax.plot([x,x],[y+sign*1.5,y+sign*4],color=point["color"],lw=.45,zorder=4)

def draw_plane(ax,mri,point,plane):
    name,u,v,labels=plane
    xyz=np.asarray(point["xyz"])
    grid,data,center=sample_native(mri,xyz,u,v,90,.5,global_context=True)
    ax.imshow(data,cmap="gray",origin="lower",extent=[-90,90,-90,90],
              vmin=mri["limits"][0],vmax=mri["limits"][1],interpolation="none")
    xy=np.array([np.dot(xyz-center,u),np.dot(xyz-center,v)])
    marker(ax,*xy,point)
    ax.set_xlim(-90,90);ax.set_ylim(-90,90);ax.set_aspect("equal");ax.axis("off")
    for label,where in zip(labels,[(.02,.5),(.98,.5),(.5,.02),(.5,.98)]):
        ax.text(*where,label,transform=ax.transAxes,color="white",fontsize=6.5,ha="center",va="center",
                bbox=dict(facecolor="black",alpha=.45,edgecolor="none",pad=.5))
    ax.plot([-78,-58],[-77,-77],color="white",lw=1)
    ax.text(-68,-85,"20 mm",color="white",ha="center",fontsize=6)
    inset=ax.inset_axes([.64,.64,.34,.34])
    small,zoom,_=sample_native(mri,xyz,u,v,12,.2)
    inset.imshow(zoom,cmap="gray",origin="lower",extent=[-12,12,-12,12],
                 vmin=mri["limits"][0],vmax=mri["limits"][1],interpolation="none")
    marker(inset,0,0,point)
    inset.set_xticks([]);inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(point["color"]);spine.set_linewidth(.6)
    inset.set_title("24 mm detail",fontsize=5.8,color="white",pad=1,
                    bbox=dict(facecolor="#222222",edgecolor="none",pad=.6))
    return name

def point_records(entry):
    flagged={r.split(":")[0] for r in entry["conflict_reason"].split(";") if ":" in r}
    names=[entry["contact_1"],"Bipolar midpoint",entry["contact_2"]]
    rows=[]
    for i,role in enumerate(["contact_1","midpoint","contact_2"]):
        flag=(role=="midpoint" and "midpoint_not_Hammers_insula" in entry["conflict_reason"]) or names[i] in flagged
        old=entry["old_midpoint_label"] if role=="midpoint" else entry[f"old_{role}_label"]
        new=entry["new_midpoint_label"] if role=="midpoint" else entry[f"new_{role}_label"]
        tissue="" if role=="midpoint" else TISSUE[entry[f"new_{role}_tissue"]]
        evidence="" if role=="midpoint" else EVIDENCE.get(entry[f"old_{role}_evidence"],entry[f"old_{role}_evidence"])
        rows.append(dict(name=names[i],role=role,xyz=entry["positions"][i],old=old,new=new,tissue=tissue,
                         evidence=evidence,flagged=flag,color=FLAG if flag else MIDPOINT if role=="midpoint" else CONTACT))
    return rows

def render_site(entry):
    matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
                                "svg.fonttype":"none","svg.hashsalt":entry["site_id"]})
    mri=load_mri(entry["MRI"])
    points=point_records(entry)
    planes=[
      ("Sagittal",np.array([0.,1.,0.]),np.array([0.,0.,1.]),["P","A","I","S"]),
      ("Coronal",np.array([1.,0.,0.]),np.array([0.,0.,1.]),["L","R","I","S"]),
      ("Axial",np.array([1.,0.,0.]),np.array([0.,1.,0.]),["L","R","P","A"])]
    fig=plt.figure(figsize=(12.6,11.25),facecolor="white")
    layout=fig.add_gridspec(3,4,left=.02,right=.99,bottom=.095,top=.88,
                           width_ratios=[1.12,1,1,1],wspace=.07,hspace=.16)
    fig.text(.025,.97,entry["site_id"].replace("_","  "),fontsize=15,weight="bold",va="top")
    fig.text(.025,.942,"Tasks: "+" | ".join(entry["tasks"]),fontsize=8,va="top")
    fig.text(.025,.919,"Native T1 MRI | upright orthogonal views through each exact point | native tkRAS coordinates (mm)",fontsize=8,va="top")
    geometry=[]
    for row,point in enumerate(points):
        xyz=np.array(point["xyz"],float)
        voxel=nib.affines.apply_affine(mri["tkr_to_voxel"],xyz)
        roundtrip=nib.affines.apply_affine(mri["voxel_to_tkr"],voxel)
        assert np.max(np.abs(roundtrip-xyz))<1e-7
        assert ((voxel>=0)&(voxel<np.array(mri["shape"]))).all()
        geometry.append(dict(site_id=entry["site_id"],subject=entry["subject"],role=point["role"],name=point["name"],
            x_mm=xyz[0],y_mm=xyz[1],z_mm=xyz[2],voxel_i=voxel[0],voxel_j=voxel[1],voxel_k=voxel[2],
            roundtrip_max_error_mm=float(max(abs(roundtrip-xyz))),MRI=entry["MRI"],flagged=point["flagged"]))
        label=fig.add_subplot(layout[row,0]);label.axis("off")
        label.text(0,.98,point["name"],fontsize=11,weight="bold",color=point["color"],va="top")
        label.text(0,.88,"REVIEW FLAG" if point["flagged"] else "Reference point",fontsize=7.5,color=point["color"],va="top")
        y=.79
        for header,value in [("OLD HAMMERS",point["old"]),("NEW HAMMERS",point["new"]),
                             ("NEW MAPER TISSUE",point["tissue"]),("OLD INCLUSION EVIDENCE",point["evidence"])]:
            if not value:continue
            label.text(0,y,header,fontsize=6.8,color="#666666",va="top");y-=.055
            lines=textwrap.wrap(value,34)
            label.text(0,y,"\n".join(lines),fontsize=8.3,va="top",linespacing=1.2)
            y-=.060*len(lines)+.065
        label.text(0,.02,"tkRAS: "+", ".join(f"{v:.3f}" for v in xyz),fontsize=7,color="#444444",va="bottom")
        for col,plane in enumerate(planes,1):
            ax=fig.add_subplot(layout[row,col]);draw_plane(ax,mri,point,plane)
            if row==0:ax.set_title(plane[0],fontsize=9,pad=5)
    handles=[Line2D([],[],marker="o",mfc="none",mec=FLAG,ls="",label="Flagged point: review pending"),
             Line2D([],[],marker="o",mfc="none",mec=CONTACT,ls="",label="Other real contact"),
             Line2D([],[],marker="D",mfc="none",mec=MIDPOINT,ls="",label="Virtual bipolar midpoint")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.052),ncol=3,fontsize=8,frameon=False)
    fig.text(.025,.034,"0 = unassigned atlas label; tissue classification is listed separately. New labels are MAPER estimates for review.",fontsize=7,color="#444444")
    fig.text(.025,.017,"Sampling membership is frozen. MRI: "+entry["MRI"],fontsize=6,color="#666666")
    IMAGES.mkdir(parents=True,exist_ok=True)
    path=IMAGES/(entry["site_id"]+"_mri_review.svg")
    fig.savefig(path,dpi=140,facecolor="white")
    plt.close(fig)
    return path,geometry

def write_gallery(entries):
    payload=[]
    for e in entries:
        p=Path(e["SVG"])
        payload.append({k:e[k] for k in ["site_id","subject","tasks","task_records","conflict_reason_zh",
            "contact_1","contact_2","old_midpoint_label","new_midpoint_label",
            "old_contact_1_label","new_contact_1_label","new_contact_1_tissue",
            "old_contact_2_label","new_contact_2_label","new_contact_2_tissue"]} |
            {"image":"data:image/svg+xml;base64,"+base64.b64encode(p.read_bytes()).decode("ascii")})
    data=json.dumps(payload,ensure_ascii=False).replace("</","<\\/")
    html="""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Insula：29 位点 MRI 核对</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f4f6;color:#18202c;font:15px system-ui,sans-serif}
header{padding:18px 24px;background:white;border-bottom:1px solid #ddd;position:sticky;top:0;z-index:4}
h1{font-size:20px;margin:0 0 10px}button,select{font:inherit;padding:7px 10px;margin:3px;border:1px solid #c5cad0;border-radius:5px;background:white}
button{cursor:pointer}button:disabled{opacity:.35}main{padding:16px 24px}.note{color:#59616c;font-size:13px}
#reason{background:#fff4e8;border-left:3px solid #d55e00;padding:10px;margin:12px 0}
#viewport{overflow:auto;background:white;border:1px solid #ddd;max-height:85vh}
#scan{display:block;width:100%;max-width:1500px;margin:auto}
table{border-collapse:collapse;width:100%;background:white;font-size:13px;margin:12px 0}
th,td{text-align:left;padding:9px;border:1px solid #ddd}#count{margin-left:8px}
</style>
<header><h1>29 个待复核位点：MRI、旧标签与新标签</h1>
<div><label>Subject <select id="subject"></select></label><label>Task <select id="task"></select></label>
<select id="site"></select><button id="prev">上一位点</button><button id="next">下一位点</button><span id="count"></span>
<button id="zoom">放大 / 适应窗口</button></div>
<div class="note">独立 MRI 核对材料 · 采样名单保持冻结 · 橙色为待复核位置，圆圈为真实触点，菱形为双极中点。每一行均为该位置自己的正交切片。</div></header>
<main><h2 id="title"></h2><div id="tasks"></div><div id="reason"></div>
<table><thead><tr><th>位置</th><th>旧 Hammers 标签</th><th>新 Hammers 标签</th><th>新组织分类</th></tr></thead><tbody id="labels"></tbody></table>
<div class="note">0 表示未获得有效灰质分区标签；组织分类另列。标签变化属于待核查证据，未自动用于删除位点。</div>
<div id="viewport"><img id="scan" alt="MRI 核对图"></div></main>
<script id="entries" type="application/json">__PAYLOAD__</script>
<script>
const entries=JSON.parse(document.getElementById('entries').textContent);
const el=id=>document.getElementById(id);let filtered=entries,index=0,zoom=false;
function option(select,value,label){let o=document.createElement('option');o.value=value;o.textContent=label;select.appendChild(o);}
option(el('subject'),'','全部 Subject');[...new Set(entries.map(e=>e.subject))].sort().forEach(s=>option(el('subject'),s,s));
option(el('task'),'','全部 Task');[...new Set(entries.flatMap(e=>e.tasks))].sort().forEach(s=>option(el('task'),s,s));
function reset(){filtered=entries.filter(e=>(!el('subject').value||e.subject===el('subject').value)&&(!el('task').value||e.tasks.includes(el('task').value)));
index=0;el('site').replaceChildren();filtered.forEach((e,i)=>option(el('site'),i,e.site_id));show();}
function show(){let e=filtered[index];el('count').textContent=filtered.length?(index+1)+' / '+filtered.length:'无匹配位点';
el('prev').disabled=index===0;el('next').disabled=index>=filtered.length-1;
if(!e){el('scan').removeAttribute('src');el('title').textContent='无匹配位点';el('labels').replaceChildren();el('tasks').textContent='';el('reason').textContent='';return;}
el('site').value=index;el('title').textContent=e.site_id;el('tasks').textContent='Tasks: '+e.tasks.join(' · ');el('reason').textContent=e.conflict_reason_zh;
el('labels').replaceChildren();[[e.contact_1,e.old_contact_1_label,e.new_contact_1_label,e.new_contact_1_tissue],
['双极中点',e.old_midpoint_label,e.new_midpoint_label,''],[e.contact_2,e.old_contact_2_label,e.new_contact_2_label,e.new_contact_2_tissue]].forEach(row=>{
let tr=document.createElement('tr');row.forEach(value=>{let td=document.createElement('td');td.textContent=value;tr.appendChild(td)});el('labels').appendChild(tr)});
el('scan').src=e.image;el('viewport').scrollTop=0;}
el('subject').onchange=reset;el('task').onchange=reset;el('site').onchange=()=>{index=Number(el('site').value);show()};
el('prev').onclick=()=>{if(index>0){index--;show()}};el('next').onclick=()=>{if(index<filtered.length-1){index++;show()}};
el('zoom').onclick=()=>{zoom=!zoom;el('scan').style.width=zoom?'1800px':'100%';el('scan').style.maxWidth=zoom?'none':'1500px'};
document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowRight')el('next').click();if(e.key==='ArrowLeft')el('prev').click()});
reset();
</script></html>"""
    path=RESULTS/"MRI_review_29_sites.html"
    path.write_text(html.replace("__PAYLOAD__",data),encoding="utf-8")
    return path

def generate_all():
    RESULTS.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((ROOT/"results/maper_restore_2026-10-18/frozen_main_inputs.json").read_text())
    assert all(sha(p)==h for p,h in baseline.items())
    entries=safe_rows();geometry=[]
    for index,entry in enumerate(entries):
        path,checks=render_site(entry);entry["SVG"]=str(path);entry["SVG_sha256"]=sha(path);geometry.extend(checks)
        print(index+1,len(entries),entry["site_id"],flush=True)
    pd.DataFrame(geometry).to_csv(RESULTS/"MRI_point_coordinate_checks.csv",index=False)
    (RESULTS/"review_manifest.json").write_text(json.dumps(entries,ensure_ascii=False,indent=2)+"\n")
    gallery=write_gallery(entries)
    assert len(entries)==29 and len(geometry)==87
    assert all(sha(p)==h for p,h in baseline.items())
    report={"completed_utc":datetime.now(timezone.utc).isoformat(),"unique_sites":len(entries),
       "subject_task_records":50,"subjects":len({e["subject"] for e in entries}),
       "native_points_checked":len(geometry),"orthogonal_panels":len(geometry)*3,
       "MRI_views":"upright native sagittal, coronal, axial through each exact point; 180mm context and 24mm detail",
       "coordinate_space":"native tkRAS mm, mapped by inverse orig.mgz vox2ras_tkr",
       "gallery":str(gallery),"gallery_self_contained":True,"SVGs":[e["SVG"] for e in entries],
       "sampling_and_NMF_unchanged":True,"protected_files_verified":len(baseline),
       "review_only":True,"scientific_export_format":"SVG"}
    (RESULTS/"validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report

