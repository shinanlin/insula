#!/usr/bin/env python3
"""Restore only D64 surfaces/annotations needed to rerender the Figure 1 example."""
import json
from pathlib import Path

from globus_cli.login_manager import LoginManager
from globus_sdk import TransferData

BASE=Path('/cwork/ns458/insula-paper')
OUT=BASE/'results/fig1/native_surface_restore'
SOURCE='bfd3e08e-22c9-11f0-9e52-0ed8719d94b5'
DEST='1ad66c7c-4f60-11e8-900c-0a6d4e044368'
SRC='/C/Users/ns458/Box/CoganLab/ECoG_Recon_Full/D64'
DST=Path('/cwork/ns458/ECoG_Recon/D64')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    client=LoginManager().get_transfer_client()
    receipt=OUT/'transfer_receipt.json'
    if receipt.exists():
        record=json.loads(receipt.read_text())
        result=client.get_task(record['task_id']).data
        print(json.dumps({k:result[k] for k in ['task_id','status','files_transferred','files_skipped','files_failed']
                          if k in result},indent=2),flush=True)
        return
    needed=[f'surf/{side}.{name}' for side in ['lh','rh'] for name in ['pial','curv']]
    needed += [f'label/{side}.aparc.a2009s.annot' for side in ['lh','rh']]
    manifests=[]
    for folder in ['surf','label']:
        listing=client.operation_ls(SOURCE,path=SRC+'/'+folder+'/').data['DATA']
        for item in listing:
            relative=folder+'/'+item['name']
            if relative in needed and not (DST/relative).exists():
                manifests.append(dict(source=SRC+'/'+relative,destination=str(DST/relative),bytes=item['size']))
    missing={p for p in needed if not (DST/p).exists()}
    assert missing=={r['source'].removeprefix(SRC+'/') for r in manifests}, 'Required source surface missing'
    (OUT/'file_manifest.json').write_text(json.dumps(manifests,indent=2)+'\n')
    if not manifests:
        print('All native example surfaces already present.',flush=True)
        return
    client.operation_ls(DEST,path=str(DST)+'/')
    transfer=TransferData(SOURCE,DEST,label='Figure 1 D64 native surfaces',sync_level='checksum',
        verify_checksum=True,encrypt_data=True,delete_destination_extra=False,
        notify_on_succeeded=False,notify_on_failed=False,notify_on_inactive=False)
    for item in manifests:transfer.add_item(item['source'],item['destination'])
    response=client.submit_transfer(transfer).data
    record=dict(task_id=response['task_id'],n_files=len(manifests),bytes=sum(r['bytes'] for r in manifests))
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
