"""Durable serial execution of the authorized roadmap experiments; no training."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/route_chain_20261107'
OUT.mkdir(parents=True, exist_ok=True)
STATUS = OUT/'status.json'
records = json.loads(STATUS.read_text()) if STATUS.exists() else {}


def run(name, arguments):
    command = [sys.executable, '-B', *arguments]
    if records.get(name, {}).get('state') == 'SUCCEEDED':
        if records[name]['command'] != command:
            raise RuntimeError('Completed stage command differs')
        return
    records[name] = {'state': 'RUNNING', 'command': command}
    STATUS.write_text(json.dumps(records, indent=2))
    tick = time.perf_counter()
    with (OUT/f'{name}.log').open('a') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    records[name].update(state='SUCCEEDED' if result.returncode == 0 else 'FAILED',
                         exit_code=result.returncode, seconds=time.perf_counter()-tick)
    STATUS.write_text(json.dumps(records, indent=2))
    print(json.dumps({'stage': name, **records[name]}), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


# Do not overlap GPU work with parameter selection.
while not (ROOT/'runs/route_sweep_20261107/complete.json').exists():
    state_path = ROOT/'logs/route_sweep_20261107.status.json'
    if state_path.exists() and json.loads(state_path.read_text())['state'] == 'FAILED':
        raise SystemExit('Preceding sweep failed; inspect it before continuing')
    time.sleep(5)
choices = 'runs/route_sweep_20261107/choices.json'
run('tests', ['-m', 'unittest', 'discover', '-s', 'lightfield_lab', '-t', '.', '-v'])
run('alignment', ['tools/verify_ta_alignment.py'])
run('depth_smoke', ['tools/route_depth.py', '--out', 'runs/route_depth_smoke_20261107', '--choices', choices, '--limit', '1', '--pupil-only'])
run('fast_smoke', ['solution/fast_inference.py', '--index', 'testing/demo_input/test.json', '--out', 'runs/route_fast_h100_20261107', '--group', 'route_fast', '--device', 'cuda'])
run('depth_baseline', ['tools/route_depth.py', '--out', 'runs/route_depth_20261107', '--choices', choices])
run('regularizer_tune', ['tools/route_depth.py', '--out', 'runs/route_regularizer_tune_20261107', '--choices', choices,
                         '--pupil-only', '--regularizers', '--subset', 'train_tune'])
run('regularizer_selection', ['tools/select_regularizer.py', '--records', 'runs/route_regularizer_tune_20261107/eyebox_records.jsonl',
                              '--out', 'runs/route_regularizer_choice_20261107.json'])
if json.loads((ROOT/'runs/route_regularizer_choice_20261107.json').read_text())['selected'] is not None:
    run('regularizer_confirmation', ['tools/route_depth.py', '--out', 'runs/route_regularizer_confirmation_20261107', '--choices', choices,
                                     '--pupil-only', '--regularizers', '--subset', 'confirmation',
                                     '--regularizer-choice', 'runs/route_regularizer_choice_20261107.json'])
run('offaxis', ['tools/route_offaxis.py', '--out', 'runs/route_offaxis_20261107'])
run('timing', ['tools/route_timing.py', '--out', 'runs/route_timing_20261107'])
(OUT/'complete.json').write_text(json.dumps({'state': 'SUCCEEDED'}))
