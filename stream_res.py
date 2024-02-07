import copy

import streamlit as st
import numpy as np
import seaborn as sns
from pickle import Unpickler
import time
import datetime
import os
from MACRO import WANDB_PATH, FILENAME_LOCAL_DUMP, FILEPATH_GLOBAL_DUMP, DICT_TABLE_INIT
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
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
        default='99999999',
    )
    parser.add_argument(
        '--filter_dataset',
        help='Filter the dataset according to the specified criteria.',
        type=str,
        default='smd',
    )

    return parser.parse_args()


st.write('### Analysis')
st.write(f'App started at {datetime.datetime.now()}')
st.write('##### 1) Load data \n'
         'Load the data about runs')
st.write('##### 2) df_view \n'
         'View a tabular summary of results')
st.write('##### 3) plot_view \n'
         'Visualize the reconstruction and the ground_truth')
st.write('##### 4) plot_metrics \n'
         'Plot some useful metrics')

start_time = time.time()
placeholder_time = st.empty()
placeholder_info = st.empty()

args = parse_args()
st.write(f'##### NOTE: runs filtered on dataset **{args.filter_dataset}**\n')


@st.cache_data()
def define_runs(WANDB_PATH, start_date, end_date):
    wandb_dir_runs = {}
    for f in sorted(WANDB_PATH.iterdir()):
        if f.is_dir() and f.name != 'run' and start_date <= f.name.split('-')[-2] <= end_date:
            wandb_dir_runs[f.name.split('-')[-1]] = f

    # wandb_dir_runs = {f.name.split('-')[-1]: f for f in WANDB_PATH.iterdir() if f.is_dir()}
    if 'run' in wandb_dir_runs.keys():
        wandb_dir_runs.pop('run')

    return wandb_dir_runs


@st.cache_data()
def update():
    st.session_state['loaded'] = True


class ProgressBytesReader():
    def __init__(self, fd, total, **kwargs):
        self.fd = fd
        self.progress = None
        self.elapsed_status = 0
        self.progress_bar = st.empty()
        self.t_start = time.time()
        self.total = total

    def read(self, size=-1):
        bytes = self.fd.read(size)
        self.progress += len(bytes)
        elapsed = time.time() - self.t_start
        if elapsed - self.elapsed_status > 3.:
            perc = self.progress / self.total
            ETA = elapsed / perc
            self.progress_bar.progress(perc,
                                       f'{self.progress // 1024 ** 2} / {self.total // 1024 ** 2} MB -- {elapsed:0.1f} / {ETA:0.1f} s')
            self.elapsed_status = elapsed
        return bytes

    def readline(self):
        bytes = self.fd.readline()
        self.progress += len(bytes)
        return bytes

    def __enter__(self):
        self.progress = 0
        return self

    def __exit__(self, *args, **kwargs):
        perc = self.progress / self.total
        self.progress_bar.progress(perc,
                                   f'{self.progress // 1024 ** 2} / {self.total // 1024 ** 2} MB -- {time.time() - self.t_start:0.3f} s')
        self.progress_bar_status = perc
        return self


# @st.cache_data()
# def load_data(filepath_dump=SAVE_ALL_DUMP):
#     print(f'Running load_data {datetime.datetime.now()}')
#
#     if filepath_dump.is_file():
#         with filepath_dump.open("rb") as fd:
#             total = os.path.getsize(filepath_dump.as_posix())
#
#             with ProgressBytesReader(fd, total=total) as pbfd:
#                 up = Unpickler(pbfd)
#                 all_dict = up.load()
#             print(f"Loaded {SAVE_ALL_DUMP.as_posix()}")
#     else:
#         raise FileNotFoundError(f'File {filepath_dump.absolute().as_posix()} not found. Please run dump_data()')
#     return all_dict

# @dataclass(frozen=True)
class Dump:

    # @st.cache_resource()
    def __init__(self, filter_dataset):
        self.filter_dataset = filter_dataset

        self.wandb_dir_runs = define_runs(WANDB_PATH, args.start_date, args.end_date)

        with FILEPATH_GLOBAL_DUMP.open('rb') as f:
            global_dict = pickle.load(f)

        self.config_dict = global_dict['config_dict']
        self.loaded_runs = list(global_dict['config_dict'].keys())
        self.list_inner_steps = global_dict['list_inner_steps']
        # print(f'self.list_inner_steps {self.list_inner_steps}')
        self.dict_table = {s: copy.deepcopy(DICT_TABLE_INIT) for s in self.list_inner_steps}
        self.list_keys_figures = {}
        self.dict_figures_feat = {}
        self.dict_vectors = {}
        self.dict_psnr_per_feat = {i: {k: {'run': [], 'psnr': [], 's0': [], 'w0': []}
                                       for k in ['train', 'val', 'test']}
                                   for i in self.list_inner_steps}
        # print(f'self.dict_psnr_per_feat {self.dict_psnr_per_feat}')
        self.progress_bar = st.sidebar.empty()
        self.elapsed_status = 0
        self.run_job_type = {'run': [], 'job': [], 'entity': [], 'train_dataset': []}
        self.load()

    def __reduce__(self):
        return self.__class__, (self.loaded_runs,)

    def _load_single(self, run):
        err_code = 0
        print('run', run)
        if self.config_dict[run]['train_dataset'] != args.filter_dataset:
            print(
                f'SKIP because of focusing on {self.filter_dataset} and this run is {self.config_dict[run]["train_dataset"]}')
            err_code = 0
            return err_code

        FILEPATH_LOCAL_DUMP = self.wandb_dir_runs[run] / 'files' / 'analysis' / FILENAME_LOCAL_DUMP
        with FILEPATH_LOCAL_DUMP.open('rb') as f:
            local_dict = pickle.load(f)

        for inner_step, table in local_dict['dict_table'].items():
            for upper_key in ['time_instance', 'windows']:
                for key_metrics, value in table[upper_key].items():
                    # st.write(inner_step, type(inner_step))
                    # st.write(self.dict_table.keys())
                    self.dict_table[inner_step][upper_key][key_metrics].extend(table[upper_key][key_metrics])

        for inner_step, table in local_dict['dict_vectors'].items():
            for split in ['train', 'val', 'test']:
                if f'psnr_{split}_per_feat' in table.keys():
                    self.dict_psnr_per_feat[inner_step][split]['run'].append(run)
                    self.dict_psnr_per_feat[inner_step][split]['psnr'].append(table[f'psnr_{split}_per_feat'])
                    # print(run)
                    # print(type(inner_step))
                    # print(self.dict_psnr_per_feat[inner_step])
                    # print(self.config_dict[run].keys())
                    # print(self.dict_figures_feat.keys())
                    if 's0' in self.config_dict[run].keys():
                        self.dict_psnr_per_feat[inner_step][split]['s0'].append(self.config_dict[run]['s0'])
                    else:
                        self.dict_psnr_per_feat[inner_step][split]['s0'].append(0.0)
                    self.dict_psnr_per_feat[inner_step][split]['w0'].append(self.config_dict[run]['w0'])

                else:
                    st.warning(f'psnr_{split}_per_feat not found in {run} for inner_step {inner_step}. Skipping...')
                    err_code = 1
                    break
            if err_code:
                break

        self.list_keys_figures[run] = local_dict['list_keys_figures']
        self.dict_figures_feat[run] = local_dict['dict_figures_feat']
        self.dict_vectors[run] = local_dict['dict_vectors']
        return err_code

    # @st.cache_data()
    def load(self):
        self.t_start = time.time()
        self.progress_bar.progress(0, f'0 / {len(self.loaded_runs)} runs')
        total = len(self.loaded_runs)
        # st.write(f'Loaded runs {self.loaded_runs}')
        # st.write(f'wandb_dir_runs {self.wandb_dir_runs.keys()}')

        for progress, run in enumerate(self.loaded_runs):
            if run in self.wandb_dir_runs.keys():
                print('run = ', run)
                err_code = self._load_single(run)
            else:
                err_code = 0
                print(f'Skip run {run} not found in wandb_dir_runs')
            elapsed = time.time() - self.t_start
            if elapsed - self.elapsed_status > 3.:
                perc = progress / total
                ETA = elapsed / perc
                self.progress_bar.progress(perc,
                                           f'{progress} / {total} -- {elapsed:0.1f} / {ETA:0.1f} s')
                self.elapsed_status = elapsed

            if err_code:
                st.warning(f'Error code {err_code} for {run}. Skipping...')
                break
        elapsed = time.time() - self.t_start
        self.progress_bar.progress(1., f'{progress} / {total} -- {elapsed:0.1f} / {elapsed:0.1f} s')

        for run, config in self.config_dict.items():
            print(config['train_dataset'], self.filter_dataset)
            if config['train_dataset'] == self.filter_dataset:
                print('equal')
                self.run_job_type['run'].append(run)
                self.run_job_type['job'].append(config['wandb_job_type'])
                self.run_job_type['entity'].append(config['entity'])
                self.run_job_type['train_dataset'].append(config['train_dataset'])
        print(self.run_job_type)

    @property
    def job_type_list(self):
        l = list(set(self.run_job_type['job']))
        l.sort()
        return l

    @property
    def dataset_type_list(self):
        l = list(set(self.run_job_type['train_dataset']))
        l.sort()
        return l

    @property
    def entity_list(self):
        l = list(set(self.run_job_type['entity']))
        l.sort()
        return l

    def __dict__(self):
        return {'dict_table': self.dict_table,
                'list_keys_figures': self.list_keys_figures,
                'dict_figures_feat': self.dict_figures_feat,
                'dict_vectors': self.dict_vectors,
                'dict_psnr_per_feat': self.dict_psnr_per_feat,
                'config_dict': self.config_dict,
                'list_inner_steps': self.list_inner_steps,
                'loaded_runs': self.loaded_runs,
                'dict_run_job_type': self.run_job_type,
                'list_jobs': self.job_type_list,
                'list_entity': self.entity_list
                }

    # st.write('### Config')
    # df_config = pd.DataFrame(config_dict)
    # st.write(df_config.transpose())


# st.sidebar.button('Overwrite_dump', on_click=overwrite_dump(st.session_state['wandb_dir_runs']))
# if 'loaded' not in st.session_state.keys():
# wandb_dir_runs = define_runs(WANDB_PATH)
#
# total = os.path.getsize(SAVE_ALL_DUMP.as_posix())

# all_dict = load_data(filepath_dump=SAVE_ALL_DUMP)
# st.write('all_dict', all_dict.keys())

placeholder_info = st.sidebar.empty()

st.cache_resource()


def load_data(args):
    dump = Dump(filter_dataset=args.filter_dataset)
    # placeholder_info.info(f'Loading data from {SAVE_ALL_DUMP.as_posix()} of {total / 1024**2:.2f} MB')
    return dump
    # st.session_state['t_load'] = time.time() - start_time


if 'loaded' not in st.session_state.keys():
    dump = load_data(args)
    placeholder_info.info(f'Loading data from {dump.loaded_runs}')

    placeholder_info.info('Loading finished')
    st.session_state.update(dump.__dict__())
    st.session_state['loaded'] = True
else:
    # st.write(dump.dict_psnr_per_feat)
    # update()
    # else:
    placeholder_info.info('Data already loaded')


# with placeholder_time.container():
# #     # if allow_log_analysis:
# #     #     st.write('Time to plot: {:.2f} seconds'.format(time.time() - t0))
#     st.write('Time to load data: {:.3f}'.format(st.session_state['t_load']))
#     # st.write('Time to end: {:.2f} seconds'.format(time.time() - start_time))
# #     # st.write('Time to load images: {:.2f} seconds'.format(end_imgs))

@st.cache_data()
def extract_ds(dict_table, dict_psnr_per_feat, dict_run_job_type):
    dict_df_time_instance = {}
    dict_df_windows = {}
    dict_df_psnr_per_feat = {}
    df_run_job_type = pd.DataFrame(dict_run_job_type).set_index('run')

    for inner_steps, v in dict_table.items():
        print(v['time_instance'])
        dict_df_time_instance[inner_steps] = pd.DataFrame(v['time_instance'])
        dict_df_windows[inner_steps] = pd.DataFrame(v['windows'])

    for inner_steps, split_table in dict_psnr_per_feat.items():
        dict_df_psnr_per_feat[inner_steps] = {}
        for split, v in split_table.items():
            if len(v['run']) > 0:
                print('inner_steps', inner_steps, 'split', split, 'v', v.keys())
                matrix = np.concatenate(v['psnr']).reshape(-1, len(v['psnr'][0]))
                dict_df_psnr_per_feat[inner_steps][split] = pd.DataFrame(matrix,
                                                                         # index=v['run']
                                                                         ).replace(0, np.nan)
                dict_df_psnr_per_feat[inner_steps][split]['run'] = v['run']
                dict_df_psnr_per_feat[inner_steps][split]['w0'] = v['w0']
                dict_df_psnr_per_feat[inner_steps][split]['s0'] = v['s0']

                dict_df_psnr_per_feat[inner_steps][split] = dict_df_psnr_per_feat[inner_steps][split].join(
                    df_run_job_type, on=['run'], how='left')
                # st.write(dict_df_psnr_per_feat[inner_steps][split])

                # print(dict_df_psnr_per_feat[inner_steps][split].index)
                dict_df_psnr_per_feat[inner_steps][split].set_index(['train_dataset', 'run', 'job', 'entity'],
                                                                    inplace=True)
                # print(dict_df_psnr_per_feat[inner_steps][split].index)
                pd.DataFrame(dict_df_psnr_per_feat[inner_steps][split]).to_csv(f'psnr_{inner_steps}_{split}.csv')
    return dict_df_time_instance, dict_df_windows, dict_df_psnr_per_feat, df_run_job_type


def make_pretty_psnr(styler):
    cm = sns.light_palette("seagreen", as_cmap=True)

    styler.background_gradient(cmap=cm, axis=1) \
        .highlight_max(color='blue', axis=1) \
        .highlight_min(color='red', axis=1) \
        .format("{:.2f}")

    return styler


# @st.cache_resource
def write_ds(dict_df_time_instance, dict_df_windows, dict_df_psnr_per_feat, df_run_job_type, inner_steps,
             selected_entity, selected_job, args):
    st.write('### Time instance')
    # for inner_steps, df in dict_df_time_instance.items():
    df = dict_df_time_instance[inner_steps]
    df.to_csv(f'df_AD_metrics_time_{args.filter_dataset}_is{inner_steps}.csv')
    st.write('#### Inner steps: {}'.format(inner_steps))
    st.write(df)

    st.write('###### Mean grouped by job type')
    time_instance_df = df.groupby('wandb_job_type').agg({
        'train_psnr': ['count', 'mean', 'std'],
        'val_psnr': ['mean', 'std'],
        'test_psnr': ['mean', 'std'],
        'auc_score': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'best_precision': ['mean'],
        'best_recall': ['mean'],
    })
    st.write(time_instance_df)
    time_instance_df.to_csv(f'df_metrics_time_{args.filter_dataset}_is{inner_steps}.csv')

    st.write('###### PSNR per feature')
    for split in ['train', 'val', 'test']:
        df = dict_df_psnr_per_feat[inner_steps][split]
        if len(selected_entity) > 0 and len(selected_job) > 0:
            df = df.loc[pd.IndexSlice[:, selected_job, selected_entity], :]
        elif len(selected_entity) > 0:
            df = df.loc[pd.IndexSlice[:, :, selected_entity], :]
        elif len(selected_job) > 0:
            df = df.loc[pd.IndexSlice[:, selected_job, :], :]
        # else:
        #     raise ValueError('No valid entity or job selected')

        st.write('{} split'.format(split))
        st.write(df.style.pipe(make_pretty_psnr))

    st.write(df_run_job_type)
    st.write('### Windows')

    # for inner_steps, df in dict_df_windows.items():
    df = dict_df_windows[inner_steps]
    st.write('#### Inner steps: {}'.format(inner_steps))
    st.write(df)
    df.to_csv(f'df_AD_metrics_windows_{args.filter_dataset}_is{inner_steps}.csv')

    st.write('Mean grouped by job type')
    windows_df = df.groupby('wandb_job_type').agg({
        'auc_score': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'best_precision': ['mean'],
        'best_recall': ['mean'],
    })
    st.write(windows_df)
    windows_df.to_csv(f'df_metrics_windows_{args.filter_dataset}_is{inner_steps}.csv')


st.sidebar.write('### Metrics showing')
selected_inner_steps_tables = st.sidebar.selectbox('Select inner steps tables', st.session_state['list_inner_steps'],
                                                   index=st.session_state['list_inner_steps'].index(3) if 3 in
                                                                                                          st.session_state[
                                                                                                              'list_inner_steps'] else 0)
selected_job_tables = st.sidebar.multiselect('Select job', st.session_state['list_jobs'])
selected_entities_tables = st.sidebar.multiselect('Select entity', st.session_state['list_entity'])

st.session_state['selected_inner_steps_tables'] = selected_inner_steps_tables
st.session_state['selected_job_tables'] = selected_job_tables
st.session_state['selected_entity_tables'] = selected_entities_tables

dict_df_time_instance, dict_df_windows, \
    dict_df_psnr_per_feat, df_run_job_type = extract_ds(dict_table=st.session_state['dict_table'],
                                                        dict_psnr_per_feat=st.session_state['dict_psnr_per_feat'],
                                                        dict_run_job_type=st.session_state['dict_run_job_type'])

write_ds(dict_df_time_instance, dict_df_windows, dict_df_psnr_per_feat, df_run_job_type,
         inner_steps=st.session_state['selected_inner_steps_tables'],
         selected_entity=st.session_state['selected_entity_tables'],
         selected_job=st.session_state['selected_job_tables'],
         args=args)

# @st.cache_data()
# def get_log_analysis_run(log_analysis_dict, selected_run):
#     print('Running get_log_analysis_run')
#     return log_analysis_dict[selected_run]


# @st.cache_data()
# def select_figures_feat(selected_inner_steps, log_analysis_run):
#     # print('keys', log_analysis_run.keys())
#     # print('selected_inner_steps', selected_inner_steps)
#     print('Running select_figures_feat')
#     dict_figures_feat = log_analysis_run[selected_inner_steps]['rec_figures']
#     list_keys_figures = list(dict_figures_feat.keys())
#     list_keys_figures.sort()
#     return list_keys_figures, dict_figures_feat

# selected_run = st.selectbox('Select run', st.session_state['runs'])
# st.session_state['selected_run'] = selected_run
# # log_analysis_run = get_log_analysis_run(st.session_state['log_analysis_dict'], selected_run)
#
# selected_inner_steps = st.selectbox('Select inner steps', st.session_state['inner_steps'], index=0)
# # list_keys_figures, dict_figures_feat = select_figures_feat(int(selected_inner_steps),log_analysis_run)
#
st.sidebar.write('### Figures showing')
selected_run = st.sidebar.selectbox('Select run', st.session_state['loaded_runs'])
st.session_state['selected_run'] = selected_run
# log_analysis_run = get_log_analysis_run(st.session_state['log_analysis_dict'], selected_run)
# print(f"list_inner_steps = {st.session_state['list_inner_steps']}")
selected_inner_steps = st.sidebar.selectbox('Select inner steps', st.session_state['list_inner_steps'],
                                            index=st.session_state['list_inner_steps'].index(3) if 3 in
                                                                                                   st.session_state[
                                                                                                       'list_inner_steps'] else 0)

# .index(st.session_state['selected_inner_steps']
#     if 'selected_inner_steps' in st.session_state.keys()
# ))
# list_keys_figures, dict_figures_feat = select_figures_feat(int(selected_inner_steps),log_analysis_run)

# st.session_state['selected_run'] = selected_run
st.session_state['selected_inner_steps'] = selected_inner_steps

# selected_run = st.session_state['selected_run']
# selected_inner_steps = st.session_state['selected_inner_steps']

# print(selected_inner_steps, type(selected_inner_steps))
# print(st.session_state['list_keys_figures'][selected_run].keys())

features = st.session_state['list_keys_figures'][selected_run][selected_inner_steps]
feats = set()
# for feat in features:
#     label, feat_num, _, _, _ = feat.split('_')
#     feat_num = feat_num.replace('feat', '')
#     feats.add(int(feat_num))

# selected_feat = st.selectbox('Select feature', list(feats), index=0)
selected_feat = st.sidebar.selectbox('Select feature',
                                     st.session_state['list_keys_figures'][selected_run][selected_inner_steps],
                                     index=st.session_state['list_keys_figures'][selected_run][
                                         selected_inner_steps].index(st.session_state['selected_feat'])
                                     if 'selected_feat' in st.session_state.keys() else 0)
st.session_state['selected_feat'] = selected_feat

choice = selected_feat.split('_')[0]
placeholder_train = st.empty()
placeholder_val = st.empty()
placeholder_test = st.empty()
# t0 = time.time()
# if allow_log_analysis:
# for label in ['train', 'val', 'test']:
#     selected_local = selected_feat.replace(choice, label)
#     # key = '_'.join(label, *choice) #feat{selected_feat}_inner_steps_{selected_inner_steps}'
#     st.plotly_chart(st.session_state['dict_figures_feat'][selected_inner_steps][selected_run][selected_local], use_container_width=True)
#### DBG
err_test = st.session_state['dict_vectors'][selected_run][selected_inner_steps]['err_test']
psnr_train_per_feat = st.session_state['dict_vectors'][selected_run][selected_inner_steps]['psnr_train_per_feat']
# th_from_val = st.session_state['dict_vectors'][selected_inner_steps][selected_run]['th_from_val']
# st.write('greater')
# st.write(np.greater(err_test, th_from_val[1]).astype(int).sum())
# st.plotly_chart(px.histogram(err_test, nbins=100))
#### DBG

# st.write(st.session_state['dict_vectors'][selected_inner_steps][selected_run]['class_point_val_feat'].sum())
# st.write(st.session_state['dict_vectors'][selected_inner_steps][selected_run]['th_from_val'])

placeholder_test.image(st.session_state['dict_figures_feat'][selected_run][selected_inner_steps]['rec'][
                           selected_feat.replace(choice, 'test')])
placeholder_val.image(st.session_state['dict_figures_feat'][selected_run][selected_inner_steps]['rec'][
                          selected_feat.replace(choice, 'val')])
placeholder_train.image(st.session_state['dict_figures_feat'][selected_run][selected_inner_steps]['rec'][
                            selected_feat.replace(choice, 'train')])

# placeholder_train.plotly_chart(st.session_state['dict_figures_feat'][selected_inner_steps][selected_run][selected_feat.replace(choice, 'train')], use_container_width=True)
# placeholder_val.plotly_chart(st.session_state['dict_figures_feat'][selected_inner_steps][selected_run][selected_feat.replace(choice, 'val')], use_container_width=True)
# placeholder_test.plotly_chart(st.session_state['dict_figures_feat'][selected_inner_steps][selected_run][selected_feat.replace(choice, 'test')], use_container_width=True)
st.image(st.session_state['dict_figures_feat'][selected_run][selected_inner_steps]['mask']['err_test'])
st.image(st.session_state['dict_figures_feat'][selected_run][selected_inner_steps]['mask']['class_point_val_feat'])
st.image(st.session_state['dict_figures_feat'][selected_run][selected_inner_steps]['mask']['class_true'])
st.image(st.session_state['dict_figures_feat'][selected_run][selected_inner_steps]['mask']['err_test_win'])
st.image(st.session_state['dict_figures_feat'][selected_run][selected_inner_steps]['mask']['class_windows_val_feat'])
st.image(st.session_state['dict_figures_feat'][selected_run][selected_inner_steps]['mask']['class_true_win'])


@st.cache_data()
def get_err(selected_inner_steps, selected_run, dict_vectors):
    err_val = dict_vectors[selected_inner_steps][selected_run]['err_val']
    err_test = dict_vectors[selected_inner_steps][selected_run]['err_test']
    return err_val, err_test


# st.write(all_dict['dict_vectors'][selected_inner_steps])
# st.write(err_val.min(), err_val.max(), np.median(err_val))
# th = st.sidebar.slider('Select threshold', min_value=err_val.min(), max_value=err_val.max(), value=err_val.max(), step=(err_val.min()-err_val.max())/100)
th = st.sidebar.slider('Select threshold', min_value=0.0, max_value=np.max(err_test), value=np.min(err_test), step=0.01)

# @st.cache_data()
# def get_mask(matrix, log_scale):
#     xe = np.arange(matrix.shape[0])
#     ye = np.arange(matrix.shape[1])
#
#     fig = go.Figure(data=go.Heatmap(
#               x = xe,
#               y = ye,
#               z = np.log10(matrix.transpose()) if log_scale else matrix.transpose(),
#               type = 'heatmap',
#               colorscale = 'Viridis'))
#
#     axis_template_x = dict(range = np.array([0,len(xe)]) - 0.5, autorange = False,
#                  showgrid = False, zeroline = False,
#                  linecolor = 'black', showticklabels = True,
#                  ticks = '' )
#
#     axis_template_y = dict(range = np.array([0,len(ye)]) - 0.5, autorange = False,
#                  showgrid = False, zeroline = False,
#                  linecolor = 'black', showticklabels = True,
#                  ticks = 'outside' )
#
#     fig.update_layout(margin = dict(t=10,r=10,b=10,l=10),
#         xaxis = axis_template_x,
#         yaxis = axis_template_y,
#         showlegend = False,
#         width = 1024,
#                       height = 768,
#         autosize = False)
#     return fig
#
# err_val, err_test = get_err(selected_inner_steps, selected_run, st.session_state['dict_vectors'])
# st.write('#### Error test')
# fig = get_mask(err_test, log_scale=False)
# # st.plotly_chart(px.imshow(err_test > th, aspect='equal'))#, binary_string=True, binary_backend='jpg').update_layout(margin=dict(l=0, r=0, t=0, b=0)))
# st.plotly_chart(fig)
# st.write('#### Error test th')
# fig = get_mask(matrix=(err_test>th).astype(float), log_scale=False)
# st.plotly_chart(fig)

# @st.cache_data()
# def get_figures(selected_run, wandb_dir_runs):
#     wandb_dir = wandb_dir_runs[selected_run]
#     analysis_dir = wandb_dir / 'files' / 'analysis'
#     list_imgs = []
#     for png_file in analysis_dir.glob('*.png'):
#         list_imgs.append(png_file.open('rb').read())
#
#     return list_imgs
# #
# list_imgs = get_figures(selected_run,wandb_dir_runs
#                         )
# st.write('#### Config')
# st.write(st.session_state['config_dict'][selected_run])
#
# for img in list_imgs:
#     st.image(img, use_column_width=True)

# end_imgs = time.time() - start_imgs