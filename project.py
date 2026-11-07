"""Single team entry point: check, report, or reproduce research experiments."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','report','research'])
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.action=='check':
        subprocess.run([sys.executable,'-m','unittest','test_portfolio','test_prediction_comparison',
                        'test_research','test_dividend_account','test_new_period'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'issuer_dividends.py','--offline'],cwd=ROOT,check=True)
    elif args.action=='report':
        cmd=[sys.executable,'build_research_evidence.py']
        if args.output:cmd+=['--output',str(args.output.resolve())]
        subprocess.run(cmd,cwd=ROOT,check=True)
    else:
        if not args.output:p.error('research requires --output pointing to a new experiment directory')
        subprocess.run([sys.executable,'run_research.py','--output',str(args.output.resolve())],cwd=ROOT,check=True)


if __name__=='__main__':main()
