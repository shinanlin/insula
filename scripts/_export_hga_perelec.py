#!/usr/bin/env python3
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pathlib import Path
from mne_bids import BIDSPath
from src.paths import RESULTS_ROOT, hga_results_dir

TASKS=["PhonemeSequence","LexicalDelay","PictureNaming","LexicalNoDelay"]
REF,ATLAS="bipolar","hammers"
DESCRIPTION,MODALITY="Repeat","sound"
EXCLUDE={"D0121"}

from src.paths import nmf_assignments_path
keep=set(pd.read_csv(nmf_assignments_path()).channel)

paths=[]
for task in TASKS:
    paths.extend(BIDSPath(root=str(hga_results_dir(task)), datatype="HGA",suffix="time",check=False).match())
frames=[]
for p in paths:
    df=pd.read_csv(p)
    if "channel" not in df.columns: continue
    df=df[df.channel.isin(keep)]
    if not df.empty: frames.append(df)
h=pd.concat(frames,ignore_index=True)
h.loc[h.phase=="Resp","phase"]="Response"; h.loc[h.phase=="Audio","phase"]="Stimulus"
h["phase"]=h.phase.astype(str).str.lower()
h=h[(h.description==DESCRIPTION)&(h.modality==MODALITY)&(~h.subject.isin(EXCLUDE))]
# per-electrode per-phase per-time mean over trials/tasks
agg=h.groupby(["channel","phase","time"],as_index=False)["value"].mean()
agg.to_csv("results/nmf/_hga_perelec_phase_time.csv", index=False)
print("OK rows",len(agg),"chans",agg.channel.nunique(),"phases",sorted(agg.phase.unique()))
