"""Re-summarize downloaded records without model weights, CUDA, or rerunning jobs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.overnight_eval import evaluation_subset, summarize, summarize_eyebox, summarize_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    found, membership = [], {}
    for name in ('offaxis', 'eyebox', 'sweep'):
        records = args.records_dir/f'{name}_records.jsonl'
        if not records.is_file():
            continue
        destination = args.out/f'{name}_summary.json'
        if name == 'offaxis':
            summarize(records, destination, 'category')
        else:
            {'eyebox': summarize_eyebox, 'sweep': summarize_sweep}[name](records, destination)
            rows = [json.loads(line) for line in records.read_text().splitlines() if line.strip()]
            membership[name] = [{'id': r['id'], 'source_id': r.get('source_id', r['id']),
                                 'split': r['split'], 'evaluation_subset': evaluation_subset(r)} for r in rows]
        found.append(name)
    if not found:
        parser.error('No offaxis/eyebox/sweep_records.jsonl found.')
    (args.out/'partition_membership.json').write_text(json.dumps(membership, indent=2), encoding='utf-8')
    print('Summarized: '+', '.join(found))


if __name__ == '__main__':
    main()
