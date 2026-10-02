"""Maintainer helper: export byte-exact external evidence; never upload it automatically."""
import argparse
import json
from pathlib import Path
import shutil
from evaluation.migration import baseline_inventory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Output must be new')
    inventory=baseline_inventory(args.baseline_root.resolve())
    args.output.mkdir(parents=True)
    for name in inventory:
        target=args.output/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.baseline_root/name,target)
    (args.output/'export_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print('Exported',len(inventory),'verified files; contains local raw records, not a public sanitized report')


if __name__=='__main__':main()
