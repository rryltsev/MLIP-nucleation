"""Reconcile DFT/kpLib and FM-baseline parent-job costs under the shared cap."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
from cluster_ssh import ROOT
from install_vasp import remote


def main():
    manifests=[json.loads(p.read_text()) for pattern in ('al32-*/manifest.json','fm-al32-baseline-*/manifest.json') for p in sorted((ROOT/'runs').glob(pattern))]
    ids=[m['job_id'] for m in manifests];assert len(set(ids))==len(ids) and all(re.fullmatch(r'\d+',j) for j in ids)
    commands=[['sacct','-n','-P','-j',','.join(ids),'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,AllocCPUS,CPUTimeRAW,AllocTRES%100,NodeList'],
        ['squeue','-h','-u','u3129','-o','%i|%j|%T']]
    code='import json,subprocess\nprint(json.dumps([subprocess.check_output(c,universal_newlines=True) for c in '+repr(commands)+']))'
    raw,queue=json.loads(remote("python3 - <<'PY'\n"+code+"\nPY"))
    records={f[0]:f for line in raw.splitlines() if len(f:=line.split('|'))>=9 and f[0] in ids}
    jobs=[]
    for m in manifests:
        f=records[m['job_id']]
        assert f[1]==m['run_id'] and (f[2] in ('COMPLETED','FAILED') or f[2].startswith('CANCELLED'))
        jobs.append(dict(job_id=f[0],run_id=f[1],state=f[2],exit_code=f[3],wall_seconds=int(f[4]),CPUs=int(f[5]),CPU_seconds=int(f[6]),allocated_TRES=f[7],node=f[8]))
    live=[l for l in queue.splitlines() if len(l.split('|'))>=3 and l.split('|')[1].startswith(('al32-','fm-al32-baseline-'))]
    assert not live
    total=sum(j['CPU_seconds'] for j in jobs);assert total<=128*3600
    record=dict(observed_utc=datetime.now(timezone.utc).isoformat(),host='umt.imm.uran.ru',jobs=jobs,live_related_jobs=live,
        total_CPU_seconds=total,total_CPU_hours=total/3600,remaining_CPU_hours=128-total/3600,raw_sacct=raw,
        interpretation='Parent CPUTimeRAW only; includes failed/cancelled attempts, no step double counting')
    target=ROOT/'runs/fm-al32-baseline-20261019-02/scheduler-all-attempts.json'
    target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('total_CPU_seconds','total_CPU_hours','remaining_CPU_hours','live_related_jobs')}))


if __name__=='__main__':main()
