"""Slurm entry for the current Figure3 cohort refresh."""
from pathlib import Path
import argparse
import os
import sys
sys.path.insert(0,str(Path('/cwork/ns458/insula-paper')))
from src import figure3_current as current

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'),'Run via Slurm'
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('audit','worker','collect','execute'))
    parser.add_argument('--index',type=int);args=parser.parse_args()
    if args.mode=='worker':current.worker(args.index)
    else:{'audit':current.audit_inputs,'collect':current.collect,'execute':current.execute_notebook}[args.mode]()
