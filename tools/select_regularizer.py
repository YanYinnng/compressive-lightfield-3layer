"""Freeze a pupil-proxy candidate using train_tune only; never an official score."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--records', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
rows = [json.loads(s) for s in args.records.read_text().splitlines() if s.strip()]
assert rows and all(r['evaluation_subset'] == 'train_tune' for r in rows)
summary = {}
for name in ['fixed08_100', 'anchor001', 'anchor01', 'smooth001', 'quantized_tail']:
    centre = np.mean([r['centre_psnr_dB'][name] for r in rows])
    pupil = np.mean([np.mean([v['psnr_dB'][name] for v in r['views']
                             if v['kind'] == 'subaperture' and np.hypot(*v['offset_mm']) > 0]) for r in rows])
    summary[name] = {'centre_dB': float(centre), 'pupil_dB': float(pupil)}
baseline = summary['fixed08_100']
eligible = [n for n, v in summary.items() if n != 'fixed08_100' and
            v['centre_dB'] >= baseline['centre_dB']-.25 and v['pupil_dB'] >= baseline['pupil_dB']+.05]
selected = max(eligible, key=lambda n: summary[n]['pupil_dB']) if eligible else None
result = {'selected': selected, 'summary': summary, 'samples': len(rows),
          'predeclared_rule': 'mean centre loss <=0.25dB and pupil gain >=0.05dB; highest pupil PSNR',
          'scope': 'research screening rule, not official grading or final submission selection'}
if args.out.exists():
    assert json.loads(args.out.read_text()) == result, 'Existing frozen choice differs'
else:
    args.out.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
