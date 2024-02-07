import copy
from main_analysis import main as main_analysis
import argparse
import traceback
from MACRO import WANDB_PATH, F1_SCORE_TH_NUM_DEFAULT
import json
import datetime

def parse_args():
    parser = argparse.ArgumentParser(prog='launcher_analysis.py')
    parser.add_argument('--overwrite', '-o', action='store_true')
    parser.add_argument('--analysis_inner_steps', '-i', nargs='+', type=int, default=[3])
    parser.add_argument(
        '--plot_metrics',
        help='Whether to plot the metrics.',
        type=int,
        default=1
    )

    parser.add_argument(
        '--limit_runs',
        '-l',
        help='Maximum number of runs to analyze.',
        type=int,
        default=2000
    )

    parser.add_argument(
        "--plotting",
        help="Whether to plot the figure with reconstruction and ground_truth.",
        type=int,
        default=1,
    )

    parser.add_argument(
        '--quantile_th',
        help='The quantile on the validation error to use as anomaly threshold.',
        type=float,
        default=0.999,
    )

    parser.add_argument(
        '--save_figures_feat',
        '-s',
        help='Whether to save the figures of the features.',
        type=int,
        default=1
    )

    parser.add_argument(
        '--f1_score_th_num',
        '-f1t',
        type=int,
        default=F1_SCORE_TH_NUM_DEFAULT,
        help='Number of thresholds to test for the f1 score.'
    )

    parser.add_argument(
        '--start_date',
        help='The start date of the analysis.',
        type=str,
        default='20200101',
    )

    parser.add_argument(
        '--end_date',
        help='The start date of the analysis.',
        type=str,
        default='9999',
    )
    return parser.parse_args()

def init(args):
    args_analysis = copy.deepcopy(args)

    wandb_dir_runs = {}
    for f in sorted(WANDB_PATH.iterdir()):
        if f.is_dir() and f.name != 'run' and args.start_date <= f.name.split('-')[-2] <= args.end_date:
            wandb_dir_runs[f.name.split('-')[-1]] = f

    # Drop the 'run' key corresponding to the symbolic last run in wandb folder
    if 'run' in wandb_dir_runs.keys():
        wandb_dir_runs.pop('run')

    return args_analysis, wandb_dir_runs

def main(wandb_dir_runs, args_analysis, args):
    str_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    runs_overwrite = []
    runs_create = []
    runs_no_err = []
    runs_err = []

    count_runs = 0
    for run, wandb_dir in wandb_dir_runs.items():
        analysis_dir = wandb_dir / 'files' / 'analysis'
        print('-' * 80)
        print(f'analisys dir: {analysis_dir}')
        args_analysis.wandb_run_id = run

        if not args_analysis.overwrite and analysis_dir.is_dir():
            print('ANALYSIS DIR already EXISTS. Skipping')
        else:
            if not analysis_dir.is_dir():
                print('ANALYSIS DIR DOES NOT EXISTS. Creating')
                runs_create.append(run)
            else:
                print('ANALYSIS DIR EXISTS. Overwriting')
                runs_overwrite.append(run)

            count_runs += 1
            try:
                main_analysis(args_analysis)
                runs_no_err.append(run)

            except FileNotFoundError as exc:
                tb = traceback.TracebackException.from_exception(exc)
                print(f'FileNotFoundError: {tb}')
                runs_err.append(run)
            except Exception as exc:
                # tb = traceback.TracebackException.from_exception(exc)
                print(f'ERROR: {traceback.format_exc()}')
                runs_err.append(run)
            if count_runs == args.limit_runs:
                break

    with open(f'run_status_analysis_{str_now}.json', 'w') as f:
        json.dump({'runs_overwrite': runs_overwrite,
                     'runs_create': runs_create,
                     'runs_no_err': runs_no_err,
                     'runs_err': runs_err}, f)
    print(f'overwritten runs: {runs_overwrite}')
    print(f'created runs: {runs_create}')
    print(f'runs with no error: {runs_no_err}')
    print(f'runs with error: {runs_err}')

if __name__ == '__main__':
    args = parse_args()
    print(args)
    args_analysis, wandb_dir_runs = init(args)
    main(wandb_dir_runs, args_analysis, args)


