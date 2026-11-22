"""Remote background supervisor with explicit exit status and durable logs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--label',required=True)
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
command=args.command[1:] if args.command[:1]==['--'] else args.command
status=ROOT/'logs'/f'{args.label}.status.json'
status.parent.mkdir(parents=True,exist_ok=True)
started=time.perf_counter()
record={'label':args.label,'state':'RUNNING','supervisor_pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
        'command':[sys.executable,'-u',*command]}
def write():
 temporary=status.with_suffix('.json.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(status)
write()
try:
 result=subprocess.run(record['command'],cwd=ROOT,stdin=subprocess.DEVNULL)
 record.update(state='SUCCEEDED' if result.returncode==0 else 'FAILED',exit_code=result.returncode)
except Exception as error:
 record.update(state='FAILED',exit_code=1,error=repr(error))
record.update(seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat());write()
raise SystemExit(record['exit_code'])
