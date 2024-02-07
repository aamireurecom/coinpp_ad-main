import itertools

import tqdm
import json
import yaml
import pickle
from MACRO import WANDB_PATH, MAX_LOGS_LOAD, FILENAME_LOCAL_DUMP, FILEPATH_GLOBAL_DUMP, DICT_TABLE_INIT, METRICS_WINDOWS, METRICS_TIME_INSTANCE
import argparse
import multiprocessing
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_logs_load', '-m', type=int, default=MAX_LOGS_LOAD)
    parser.add_argument('--resolution_height', '-rh', type=int, default=int(1080/2))
    parser.add_argument('--resolution_width', '-rw', type=int, default=int(1920/2))
    parser.add_argument('--no_plotting', '-np', action='store_true')
    parser.add_argument('--overwrite', '-o', action='store_true')
    parser.add_argument('--wandb_run_id', '-r', type=str, default=None)
    parser.add_argument('--start_date',help='The start date of the analysis.',type=str, default='20200101')
    parser.add_argument('--end_date',help='The start date of the analysis.',type=str,default='999999',)

    return parser.parse_args()

def define_runs(WANDB_PATH, start_date, end_date):
    wandb_dir_runs = {}
    for f in sorted(WANDB_PATH.iterdir()):
        if f.is_dir() and f.name != 'run' and start_date <= f.name.split('-')[-2] <= end_date:
            wandb_dir_runs[f.name.split('-')[-1]] = f

    if 'run' in wandb_dir_runs.keys():
        wandb_dir_runs.pop('run')
    return wandb_dir_runs

def fun2img(k, v, resolution_width=1920, resolution_height=1080):
    print(k)
    return (k, v.to_image(format='png', height=resolution_width, width=resolution_height))

def fun2img_mask(k, v, resolution_width=1024, resolution_height=768):
    print(k)
    return (k, v.to_image(format='png', height=resolution_width, width=resolution_height))

def key_sort_list_keys_figures(v):
    return int(v.split('_')[1].replace('feat', ''))

def extract_log_analysis(log_analysis_file, metrics, args, pool, count_runs, flag_load_warning, no_plotting=False):
    dict_figures_feat, dict_vectors, list_keys_figures = {}, {}, {}

    with log_analysis_file.open('rb') as f:
        log_analysis = pickle.load(f)
    # log_analysis_dict[run] = log_analysis

    for inner_steps, _ in metrics.items():
        inner_steps = int(inner_steps)

        dict_vectors[inner_steps] = {}

        if not no_plotting:
            if count_runs < args.max_logs_load:
                dict_figures_feat[inner_steps] = {}
                list_keys_figures[inner_steps] = {}

                items = list((k, v, args.resolution_height, args.resolution_width)
                             for k, v in log_analysis[inner_steps]['figures']['rec'].items())

                dict_figures_feat[inner_steps] = {'rec': dict(pool.starmap(fun2img, items))}

                items = list((k, v, args.resolution_height, args.resolution_width)
                             for k, v in log_analysis[inner_steps]['figures']['mask'].items())

                dict_figures_feat[inner_steps]['mask'] = dict(pool.starmap(fun2img, items))
                # dict_figures_feat[inner_steps][run] = log_analysis[inner_steps]['rec_figures'].to_image(format='png', width=1920*4, height=1080*4)
                list_keys_figures[inner_steps] = list(dict_figures_feat[inner_steps]['rec'].keys())
                list_keys_figures[inner_steps].sort(key=key_sort_list_keys_figures)

            else:
                if not flag_load_warning:
                    print(f'Too many logs to load, reached max of {args.max_logs_load}. Skipping...')
                    flag_load_warning = True

        for k in log_analysis[inner_steps].keys():
            if k != 'figures':
                dict_vectors[inner_steps][k] = log_analysis[inner_steps][k]

    return list_keys_figures, dict_figures_feat, dict_vectors, flag_load_warning

def extract_metrics(metrics, config_args, run):
    dict_table_local = dict()
    inner_steps_local_set = set([int(i) for i in metrics.keys()])

    for inner_steps, v in metrics.items():
        inner_steps = int(inner_steps)
        dict_table_local[inner_steps] = copy.deepcopy(DICT_TABLE_INIT)

        dict_table_local[inner_steps]['time_instance']['run'].append(run)
        dict_table_local[inner_steps]['windows']['run'].append(run)

        for id in ['entity', 'wandb_job_type', 'train_dataset']:
            dict_table_local[inner_steps]['time_instance'][id].append(config_args[id])
            dict_table_local[inner_steps]['windows'][id].append(config_args[id])

        # print(f"metric_dict = {v['time_instance']}")

        for metric in METRICS_TIME_INSTANCE:
            dict_table_local[inner_steps]['time_instance'][metric].append(v['time_instance'][metric])
        for metric in METRICS_WINDOWS:
            dict_table_local[inner_steps]['windows'][metric].append(v['windows'][metric])

        # for label in ['train', 'val', 'test']:
        #     dict_table_local[inner_steps]['time_instance'][f"{label}_psnr"].append(v['time_instance'][f"{label}_psnr"])

    return dict_table_local, inner_steps_local_set

def dump_data(wandb_dir_runs, args):
    if FILEPATH_GLOBAL_DUMP.exists():
        with FILEPATH_GLOBAL_DUMP.open('rb') as f:
            d = pickle.load(f)
        config_dict = d['config_dict']
        set_inner_steps = set(d['list_inner_steps'])
    else:
        config_dict = {}
        set_inner_steps = set()

    counter_logs = 0
    max_progress = len(wandb_dir_runs)
    # placeholder_info = st.empty()
    # placeholder_progress = st.empty()
    # placeholder_progress.progress(0 / max_progress, f'Loading... 0/{max_progress}')
    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 2))

    flag_load_warning = False

    for i, (run, wandb_dir) in enumerate(tqdm.tqdm(wandb_dir_runs.items(), total=max_progress)):
        if args.wandb_run_id is not None:
            if run != args.wandb_run_id:
                continue

        analysis_dir = wandb_dir / 'files' / 'analysis'

        if analysis_dir.is_dir() and FILENAME_LOCAL_DUMP not in analysis_dir.iterdir():
            print(analysis_dir)

            metrics_file = analysis_dir / 'metrics.json'
            config_args_file = wandb_dir / 'files' / 'config_args.yaml'
            log_analysis_file = analysis_dir / 'log_analysis.pkl'
            local_dump_file = analysis_dir / FILENAME_LOCAL_DUMP

            if metrics_file.is_file() and config_args_file.is_file() and log_analysis_file.is_file():
                # runs.append(run)
                if args.overwrite or not local_dump_file.exists():
                    with metrics_file.open('r') as f:
                        metrics = json.load(f)
                    with config_args_file.open('r') as f:
                        config_args = yaml.load(f, Loader=yaml.FullLoader)
                    config_dict[run] = config_args

                    list_keys_figures, dict_figures_feat, \
                            dict_vectors, flag_load_warning = extract_log_analysis(log_analysis_file, metrics, args, pool,
                                                                                   counter_logs, flag_load_warning, no_plotting=args.no_plotting)
                    counter_logs += 1
                    dict_table, inner_steps_local_set = extract_metrics(metrics, config_args, run)
                    set_inner_steps = set_inner_steps.union(inner_steps_local_set)

                    all_dict = {
                            'dict_table': dict_table,
                            'dict_vectors': dict_vectors,
                            'dict_figures_feat': dict_figures_feat,
                            'list_keys_figures': list_keys_figures
                        }

                    with (analysis_dir / FILENAME_LOCAL_DUMP).open('wb') as f:
                        pickle.dump(all_dict, f)
                else:
                    print(f'Skipping {wandb_dir} because dump files exists. Use --overwrite to overwrite existing files.')
            else:
                print(f'SKIPPING {wandb_dir}.\nNo metrics.json or config_args.yaml file')
        else:
            print(f'SKIPPING {wandb_dir}.\nNo analysis directory')
        # placeholder_progress.progress(i/max_progress)
        # placeholder_info.info(f'Loading... {i}/{max_progress}')

    set_inner_steps = list(set_inner_steps)
    set_inner_steps.sort()

    with FILEPATH_GLOBAL_DUMP.open('wb') as f:
        pickle.dump({'config_dict': config_dict,
                     'list_inner_steps': set_inner_steps}, f)

if __name__ == '__main__':
    args = parse_args()
    wandb_dir_runs = define_runs(WANDB_PATH, args.start_date, args.end_date)
    dump_data(wandb_dir_runs, args)
