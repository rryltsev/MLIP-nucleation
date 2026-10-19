"""Snapshot all recorded Al32 jobs, rejecting untracked/live/unknown charges."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess

from install_vasp import remote

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifests=[json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('al32-*/manifest.json'))]
    ids=[m['job_id'] for m in manifests]
    assert all(re.fullmatch(r'\d+',j) for j in ids) and len(set(ids))==len(ids)
    commands=[['sacct','-n','-P','-j',','.join(ids),'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,AllocCPUS,CPUTimeRAW,NodeList'],
              ['squeue','-h','-u','u3129','-o','%i|%j|%T'],
              ['sacct','-n','-P','-u','u3129','--starttime=2026-10-18','--format=JobIDRaw,JobName%100,State']]
    code='import json,subprocess\nprint(json.dumps([subprocess.check_output(c,universal_newlines=True) for c in '+repr(commands)+']))'
    raw,queue,observed=json.loads(remote("python3 - <<'PY'\n"+code+"\nPY"))
    for line in (queue+observed).splitlines():
        f=line.split('|')
        if len(f)>=3 and f[0].isdigit() and f[1].startswith('al32-'):assert f[0] in ids,'Unrecorded Al32 job'
    accounting={f[0]:f for line in raw.splitlines() if len(f:=line.split('|'))>=8 and f[0] in ids}
    jobs=[]
    for m in manifests:
        f=accounting[m['job_id']]
        assert f[1]==m['run_id'] and f[2] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')
        assert m['scheduler_record'][:6]==[f[0]]+f[2:7]
        jobs.append(dict(job_id=f[0],run_id=f[1],state=f[2],exit_code=f[3],wall_seconds=int(f[4]),
                         allocated_cpus=int(f[5]),allocated_CPU_seconds=int(f[6]),nodes=f[7]))
    live=[l for l in queue.splitlines() if len(l.split('|'))>=3 and l.split('|')[1].startswith('al32-')]
    assert not live
    total=sum(j['allocated_CPU_seconds'] for j in jobs);assert total<=128*3600
    record=dict(observed_utc=datetime.now(timezone.utc).isoformat(),host='umt.imm.uran.ru',jobs=jobs,
        total_CPU_seconds=total,total_CPU_hours=total/3600,remaining_CPU_hours=128-total/3600,
        live_al32_jobs=live,raw_sacct=raw,raw_squeue=queue,
        interpretation='Allocated CPUTimeRAW parent jobs only; batch/extern/MPI steps excluded')
    target=ROOT/'runs/al32-kplib-benchmark-20261019-01/scheduler-final.json'
    target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('total_CPU_hours','remaining_CPU_hours','live_al32_jobs')}))


if __name__=='__main__':main()
