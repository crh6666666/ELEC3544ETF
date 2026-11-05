"""Run one model or all three using only this folder's local data."""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ('ridge', 'random_forest', 'xgboost')


def main():
    parser = argparse.ArgumentParser(description='Offline ETF model training and illustrative portfolio chart.')
    parser.add_argument('--model', choices=(*MODEL_NAMES, 'all'), default='all')
    parser.add_argument('--output', type=Path, default=ROOT/'results',
                        help='Parent folder for per-model results.')
    parser.add_argument('--cost-bps', type=float, default=10.0,
                        help='Hypothetical one-way portfolio trading cost in bps.')
    parser.add_argument('--no-plot', action='store_true', help='Export models and tables without figures.')
    args = parser.parse_args()
    if not 0 <= args.cost_bps < 10000:
        parser.error('--cost-bps must be between 0 and 10000 (exclusive).')
    out = args.output.resolve()
    protected = [ROOT/p for p in ('data', 'prepared_data', 'models', '.venv')]
    if out == ROOT or any(out.is_relative_to(p) or p.is_relative_to(out) for p in protected):
        parser.error('Choose a separate output folder such as results or experiments/trial01.')
    names = MODEL_NAMES if args.model == 'all' else (args.model,)
    metrics = []
    for name in names:
        subprocess.run([sys.executable, '-m', f'models.{name}', '--output', str(out/name)],
                       cwd=ROOT, check=True)
        metrics.append(pd.read_csv(out/name/'metrics.csv'))
    # A partial run must not replace the comparison of all three models.
    filename = 'model_comparison_metrics.csv' if args.model == 'all' else f'{args.model}_metrics.csv'
    pd.concat(metrics).to_csv(out/filename, index=False)
    print(f'Metrics: {out/filename}')
    figures = ROOT/'figures' if out == ROOT/'results' else out/'figures'
    comparison = [sys.executable, str(ROOT/'compare_predictions.py'),
                  '--results', str(out), '--figures', str(figures), '--models', *names]
    if args.no_plot:
        comparison.append('--no-plot')
    subprocess.run(comparison, cwd=ROOT, check=True)
    if not args.no_plot:
        subprocess.run([sys.executable, str(ROOT/'plot_portfolio.py'),
                        '--results', str(out), '--output', str(figures),
                        '--cost-bps', str(args.cost_bps), '--models', *names],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
