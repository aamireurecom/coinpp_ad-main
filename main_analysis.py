import argparse
import itertools
import pathlib
import yaml
import helpers
import torch
from coinpp.training import Trainer
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score

from affiliation import pr_from_events
from affiliation.generics import convert_vector_to_events, f1_func

import matplotlib.pyplot as plt
import json
import pickle
import tempfile
import shutil
import os
import plotly.graph_objects as go
from evaluation_utils import get_events, get_point_adjust_scores, get_composite_fscore_raw
from MACRO import WANDB_PATH, F1_SCORE_TH_NUM_DEFAULT
import multiprocessing


def mse2psnr(mse):
    return -10. * np.log10(mse + 10e-20)


def parse_args():
    parser = argparse.ArgumentParser(prog='main_analysis.py')

    parser.add_argument(
        "--analysis_inner_steps",
        help="List of number of inner loop steps for the reconstruction. It will run the actual analysis for each value in the list.",
        type=int,
        nargs="+",
        default=[3],
    )

    parser.add_argument(
        "--plotting",
        help="Whether to plot the figure with reconstruction and ground_truth.",
        type=int,
        default=1,
    )

    parser.add_argument(
        '--overwrite',
        '-o',
        help='Whether to overwrite the existing analysis.',
        action='store_true',
    )

    parser.add_argument(
        '--plot_metrics',
        help='Whether to plot and save a group of metrics for debugging purposes. Tested only for SMD dataset',
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
        '--wandb_run_id',
        '-r',
        type=str,
        default='pr7woft5',
        help='Wandb run id to load the model from. Please provide the hash code of the run uniquely defined by wandb.'
    )

    parser.add_argument(
        '--save_figures_feat',
        '-s',
        action='store_true',
        help='Whether to save the figures representing each separate feature of the time-series, with its own reconstruction, ground truth,' 
             'label and error.'
    )

    parser.add_argument(
        '--f1_score_th_num',
        '-f1t',
        type=int,
        default=F1_SCORE_TH_NUM_DEFAULT,
        help='Number of thresholds to test for the f1 score.'
    )

    return parser.parse_args()


def load_model_config(args):
    """
        Load the model configuration and related files for analysis.

        Args:
            args: An object containing command-line arguments and options.

        Returns:
            model_state_dict: The state dictionary of the loaded model.
            args_train: The training configuration as an argparse.Namespace object.
            wandb_dir: The directory path for the WandB run.
            dirpath_figures: The directory path for storing analysis figures.
            model_epoch: The epoch of the loaded model.

        Raises:
            FileExistsError: If the model file or configuration file is not found.

    """

    wandb_dir_runs = {f.name.split('-')[-1]: f for f in WANDB_PATH.iterdir() if f.is_dir()}
    # print('wandb_dir_runs', wandb_dir_runs.keys())
    wandb_dir = wandb_dir_runs[args.wandb_run_id]

    list_models = list((wandb_dir / 'files').glob('model*'))
    if len(list_models) == 0:
        raise FileExistsError(f'Could not find a model "model*.pt in {wandb_dir}')

    # Load last model, ordered by epoch
    list_models.sort()
    model_path = list_models[-1]

    print('Loading model from', model_path)
    model_epoch = int(model_path.stem.split('_')[1].replace('epoch', '')[-1])
    if model_path.exists():
        dict_load = torch.load(model_path, map_location=torch.device('cpu'))
        model_state_dict = dict_load['state_dict']
    else:
        raise FileExistsError(f'Could not find the looked model.pt in {wandb_dir}')

    config_train_path = wandb_dir / 'files' / 'config_args.yaml'
    print('Loading config from', config_train_path)

    if config_train_path.exists():
        with config_train_path.open('r') as f:
            config_train = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise FileExistsError(f'Could not find config_args.yaml in {wandb_dir}')

    dirpath_figures = (wandb_dir / 'files' / 'analysis')

    if args.overwrite:
        if dirpath_figures.exists():
            tmp = tempfile.mktemp(dir=os.path.dirname(dirpath_figures.as_posix()))
            # Rename the dir.
            shutil.move(dirpath_figures.as_posix(), tmp)
            # And delete it.
            shutil.rmtree(tmp)

    dirpath_figures.mkdir(parents=False, exist_ok=args.overwrite)

    # Convert config_train to argparse.Namespace
    print('Defining args_train')

    # Modify parameters device and full_anomaly_detection for legacy analysis
    config_train['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'full_anomaly_detection' not in config_train.keys():
        config_train['full_anomaly_detection'] = 1

    args_train = argparse.Namespace(**config_train)

    return model_state_dict, args_train, wandb_dir, dirpath_figures, model_epoch


def save_figures(log_dict, root_dir, inner_steps, selected_features):
    dict_figures = {}
    for label in ['train', 'test', 'val']:
        for i in selected_features:
            dict_figures[f'{label}_feat{i}_inner_steps_{inner_steps}'] = log_dict[label][
                f'{label}_reconstruction_{inner_steps}_steps_feat-{i}']
    return dict_figures


def run_loop(args, model_state_dict, args_train, dirpath_figures):
    """
        Run the analysis loop using the provided arguments, model state dictionary, training configuration, and figures directory.

        Args:
            args: An object containing command-line arguments and options.
            model_state_dict: The state dictionary of the loaded model.
            args_train: The training configuration as an argparse.Namespace object.
            dirpath_figures: The directory path for storing analysis figures.

        Returns:
            log_dict: A dictionary containing the analysis results and trainer information.

    """
    # Build datasets, converters and model
    train_dataset, test_dataset, converter = helpers.get_datasets_and_converter(args_train)
    model = helpers.get_model(args_train)
    print(model)
    print('model_state_dict', model_state_dict.keys())
    model.load_state_dict(model_state_dict)

    print(f'Args \n {args}')
    print(f'Args_train \n {args_train}')

    # Initialize trainer to validate model
    trainer = Trainer(
        func_rep=model,
        converter=converter,
        args=args_train,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_path="",
    )

    # Run the inner loop for the many number of inner steps defined in args.analysis_inner_steps
    log_dict = {'trainer': trainer}
    for inner_steps in args.analysis_inner_steps:
        print(f'Running analysis with inner_steps={inner_steps}')
        inner_log = {}
        log_dict[inner_steps] = inner_log

        # Run validation over the different splits
        for label, dataloader in zip(['train', 'test', 'val'],
                                     [trainer.train_monitor_dataloader, trainer.test_dataloader,
                                      trainer.val_dataloader]):
            print(f'Running analysis on {label}')
            inner_log[label] = trainer.analyse(dataloader, label=label, inner_steps=inner_steps, plotting=args.plotting)

        if args.save_figures_feat:
            dict_figures = save_figures(log_dict=inner_log, root_dir=dirpath_figures, inner_steps=inner_steps,
                                        selected_features=trainer.selected_features)
            inner_log['rec_figures'] = dict_figures

    log_dict['trainer'] = trainer
    return log_dict


def plot_err(err_test_norm2, class_true, title, savename, dirpath, saving=True):
    """
    Plot the error (anomaly score) of the test set, with the true class as color.
    """
    x = np.arange(len(err_test_norm2))
    x0 = x[class_true == 0]
    x1 = x[class_true == 1]

    plt.figure()
    plt.scatter(x0, err_test_norm2[class_true == 0], c='b', s=1)
    plt.scatter(x1, err_test_norm2[class_true == 1], c='r', s=1)
    plt.title(title)
    if saving:
        plt.savefig(dirpath / f'{savename}')
    plt.close()


def plot_err_distribution(err_test_norm2, class_true, title, savename, dirpath, saving=True):
    """
    Plot the error distribution (anomaly score) of the test set, with the true class as color.
    """
    plt.figure()
    plt.hist(err_test_norm2[class_true == 0], bins=100, density=True, alpha=0.5, label='Normal')
    plt.hist(err_test_norm2[class_true == 1], bins=100, density=True, alpha=0.5, label='Anomaly')
    plt.title(title)
    plt.legend()
    if saving:
        plt.savefig(dirpath / f'{savename}')
    plt.close()


def plot_roc(fpr, tpr, savename, title, dirpath, saving=True):
    """
    Plot the ROC curve.
    """
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(title)
    plt.grid('on')
    if saving:
        plt.savefig(dirpath / f'{savename}')
    plt.close()


def plot_f1_th(f1, th, th_val, savename, title, dirpath, saving=True):
    """
    Plot the chosen threshold of the F1 score.
    """
    plt.figure()
    plt.plot(th, f1)
    plt.vlines(th_val, 0, 1, linestyles='dashed', colors='r')
    plt.title(title)
    plt.grid('on')
    if saving:
        plt.savefig(dirpath / f'{savename}')
    plt.close()


def get_mask(matrix, title, log_scale=True):
    """
    Plot the heatmap mask of the time-series matrix to highlight the anomalous features.
    """
    xe = np.arange(matrix.shape[0])
    ye = np.arange(matrix.shape[1])

    fig = go.Figure(data=go.Heatmap(
        x=xe,
        y=ye,
        z=np.log10(matrix.transpose()) if log_scale else matrix.transpose(),
        type='heatmap',
        colorscale='Viridis'))

    axis_template_x = dict(range=np.array([0, len(xe)]) - 0.5, autorange=False,
                           showgrid=False, zeroline=False,
                           linecolor='black', showticklabels=True,
                           ticks='')

    axis_template_y = dict(range=np.array([0, len(ye)]) - 0.5, autorange=False,
                           showgrid=True, zeroline=False,
                           linecolor='black', showticklabels=True,
                           ticks='outside')

    fig.update_layout(margin=dict(t=50, r=20, b=20, l=20),
                      xaxis=axis_template_x,
                      yaxis=axis_template_y,
                      showlegend=False,
                      width=768,
                      height=1024,
                      autosize=False,
                      title=title)
    return fig


def analyse_log(args, args_train, log_dict, dirpath_analysis, model_epoch):
    """
    Parameters:

    args: Arguments for analysis
    args_train: Training arguments
    log_dict: Dictionary containing log information
    dirpath_analysis: Directory path for analysis
    model_epoch: Epoch of the model

    Description:
    This function performs analysis on the log data provided. It calculates various metrics and generates visualizations for the analysis.

    The function starts by initializing some variables and extracting necessary information from the log_dict dictionary.

    It then iterates over the args.analysis_inner_steps list and performs the following steps for each inner step:

    Extracts the relevant log information for the current inner step.
    Calculates error metrics and PSNR (Peak Signal-to-Noise Ratio) values for the training, validation, and test data.
    Computes threshold values based on quantiles of the validation error.
    Performs classification based on the threshold values to determine the class labels for each data point and window.
    Calculates metrics such as AUC (Area Under the Curve), F1 score, precision, and recall for both time instance and window-based classifications.
    Stores the calculated metrics and other analysis data in the log_analysis dictionary.
    The function also includes some print statements to display information during the analysis.

    If the args.save_figures_feat argument is set to True, the function generates and stores figures related to the analysis, such as error plots, class masks, and more.

    """
    metadata = {'epoch': model_epoch}
    metrics = {}
    log_analysis = {}

    # Extract relevant log information
    class_true = log_dict['trainer'].test_dataset.labels
    class_true_win = log_dict['trainer'].test_dataset.labels_windows

    selected_features = log_dict['trainer'].selected_features
    num_tot_features = log_dict['trainer'].train_dataset.num_tot_features
    num_features = len(selected_features)

    true_events_single_COM = log_dict['trainer'].test_dataset.true_events_single_COM
    true_events_win_COM = log_dict['trainer'].test_dataset.true_events_win_COM

    true_events_single_AFF = log_dict['trainer'].test_dataset.true_events_single_AFF
    true_events_win_AFF = log_dict['trainer'].test_dataset.true_events_win_AFF

    for inner_steps in args.analysis_inner_steps:
        # Compute error metrics for the current inner step
        print('-' * 30)
        print(f'inner_steps {inner_steps}')
        inner_log = log_dict[inner_steps]

        metrics[inner_steps] = {}

        y_train = inner_log['train'][f'y_feat']
        yhat_train = inner_log['train'][f'yhat_feat']

        # err_train = np.sqrt(((yhat_train - y_train) ** 2))
        # th_from_train = np.quantile(err_train, 0.999, axis=0)

        yhat_val = inner_log['val'][f'yhat_feat']
        y_val = inner_log['val'][f'y_feat']

        yhat_test = inner_log['test'][f'yhat_feat']
        y_test = inner_log['test'][f'y_feat']

        err_train = np.sqrt((yhat_train - y_train) ** 2)
        psnr_train_per_feat = np.zeros(num_tot_features)
        psnr_train_per_feat[selected_features] = -10 * np.log10((err_train ** 2).mean(axis=0))

        err_val = np.sqrt((yhat_val - y_val) ** 2)
        err_val_win = err_val.reshape(-1, args_train.window_length[0], num_features).mean(axis=1)
        psnr_val_per_feat = np.zeros(num_tot_features)
        psnr_val_per_feat[selected_features] = -10 * np.log10((err_val ** 2).mean(axis=0))

        err_val_norm2 = np.linalg.norm(err_val, axis=1)
        err_val_norm2_win = err_val_norm2.reshape(-1, args_train.window_length[0]).mean(axis=1)

        err_test = np.sqrt((yhat_test - y_test) ** 2)

        err_test_win = err_test.reshape(-1, args_train.window_length[0], num_features).mean(axis=1)

        psnr_test_per_feat = np.zeros(num_tot_features)
        psnr_test_per_feat[selected_features] = -10 * np.log10((err_test ** 2).mean(axis=0))

        err_test_norm2 = np.linalg.norm(err_test, axis=1)
        err_test_norm2_win = err_test_norm2.reshape(-1, args_train.window_length[0]).mean(axis=1)

        # for quantile in args.quantile_th:
        th_from_val = np.quantile(err_val, args.quantile_th, axis=0)
        th_from_val_norm2 = np.quantile(err_val_norm2, args.quantile_th, axis=0)
        th_from_val_norm2_win = np.quantile(err_val_norm2_win, args.quantile_th)

        class_point_val_feat = np.greater(err_test, th_from_val).astype(int)
        class_point_val = np.any(class_point_val_feat, axis=1)

        class_windows_val_feat = np.any(class_point_val_feat.reshape(-1, num_features, args_train.window_length[0]),
                                        axis=2)
        class_windows_val = np.any(class_windows_val_feat, axis=1)

        # Instantiate metrics dictionary
        metrics_time_instance = {}
        metrics_windows = {}

        # ---------------------------------------------------------------------------
        # Compute metrics for time instances
        fpr, tpr, ths = roc_curve(class_true, err_test_norm2)
        auc_single = roc_auc_score(class_true, err_test_norm2)

        list_f1_single = []
        list_f1PA_single = []
        list_f1COM_single = []
        list_f1AFF_single = []

        TrangeAFF = (0, len(err_test_norm2))
        ths_single = np.linspace(np.min(err_test_norm2), np.max(err_test_norm2), args.f1_score_th_num)

        for th in ths_single:
            pred_labels = np.greater(err_test_norm2, th).astype(int)
            pred_events = convert_vector_to_events(pred_labels)
            list_f1_single.append(f1_score(class_true, pred_labels))
            list_f1PA_single.append(
                get_point_adjust_scores(y_test=class_true, pred_labels=pred_labels, true_events=true_events_single_COM)[-1])
            list_f1COM_single.append(
                get_composite_fscore_raw(y_test=class_true, pred_labels=pred_labels, true_events=true_events_single_COM))
            AFF_PR = pr_from_events(events_pred=pred_events, events_gt=true_events_single_AFF, Trange=TrangeAFF)
            list_f1AFF_single.append(f1_func(np.mean(AFF_PR['individual_precision_probabilities']), np.mean(AFF_PR['individual_recall_probabilities'])))

        best_th_single = ths_single[np.argmax(list_f1_single)]
        pred_labels_best = np.greater(err_test_norm2, best_th_single).astype(int)
        best_precision_single = precision_score(class_true, pred_labels_best)
        best_recall_single = recall_score(class_true, pred_labels_best)
        best_f1PA_single = max(list_f1PA_single)
        best_f1AFF_single = max(list_f1AFF_single)

        best_f1_single = max(list_f1_single)
        fpr_win, tpr_win, ths_win = roc_curve(class_true_win, err_test_norm2_win)
        auc_win = roc_auc_score(class_true_win, err_test_norm2_win)

        # ---------------------------------------------------------------------------
        # Compute metrics for windows

        list_f1_win = []
        list_f1PA_win = []
        list_f1COM_win = []
        list_f1AFF_win = []
        ths_win = np.linspace(np.min(err_test_norm2_win), np.max(err_test_norm2_win), args.f1_score_th_num)
        TrangeAFF = (0, len(err_test_norm2_win))

        for th in ths_win:
            pred_labels = np.greater(err_test_norm2_win, th).astype(int)
            pred_events = convert_vector_to_events(pred_labels)

            list_f1_win.append(f1_score(class_true_win, pred_labels))
            list_f1PA_win.append(
                get_point_adjust_scores(y_test=class_true_win, pred_labels=pred_labels, true_events=true_events_win_COM)[
                    -1])
            list_f1COM_win.append(
                get_composite_fscore_raw(y_test=class_true_win, pred_labels=pred_labels, true_events=true_events_win_COM))
            AFF_PR = pr_from_events(events_pred=pred_events, events_gt=true_events_win_AFF, Trange=TrangeAFF)
            list_f1AFF_win.append(f1_func(np.mean(AFF_PR['individual_precision_probabilities']), np.mean(AFF_PR['individual_recall_probabilities'])))

        best_th_win = ths_win[np.argmax(list_f1_win)]
        best_precision_win = precision_score(class_true_win, np.greater(err_test_norm2_win, best_th_win).astype(int))
        best_recall_win = recall_score(class_true_win, np.greater(err_test_norm2_win, best_th_win).astype(int))
        best_f1PA_win = max(list_f1PA_win)
        best_f1AFF_win = max(list_f1AFF_win)

        print(f'max_single = {max(list_f1_single)}')
        print(f'max_single = {max(list_f1_win)}')

        metrics_time_instance['auc_score'] = auc_single
        metrics_time_instance['f1_score'] = f1_score(class_true, class_point_val)
        metrics_time_instance['best_f1_score'] = best_f1_single
        metrics_time_instance['best_f1PA_score'] = best_f1PA_single
        metrics_time_instance['best_f1COM_score'] = max(list_f1COM_single)
        metrics_time_instance['best_f1AFF_score'] = best_f1AFF_single
        metrics_time_instance['best_precision'] = best_precision_single
        metrics_time_instance['best_recall'] = best_recall_single
        # metrics_time_instance['best_f1_score_th'] = ths[np.argmax(list_f1_score_single)]
        metrics_time_instance['precision'] = precision_score(class_true, class_point_val)
        metrics_time_instance['recall'] = recall_score(class_true, class_point_val)

        for label in ['train', 'val', 'test']:
            metrics_time_instance[f'{label}_psnr'] = inner_log[label][f"{label}_psnr_{inner_steps}_steps"]

        metrics_windows['auc_score'] = auc_win
        metrics_windows['f1_score'] = f1_score(class_true_win, class_windows_val)
        metrics_windows['best_f1_score'] = max(list_f1_win)
        metrics_windows['best_f1PA_score'] = best_f1PA_win
        metrics_windows['best_f1COM_score'] = max(list_f1COM_win)
        metrics_windows['best_f1AFF_score'] = best_f1AFF_win
        metrics_windows['best_precision'] = best_precision_win
        metrics_windows['best_recall'] = best_recall_win
        metrics_windows['precision'] = precision_score(class_true_win, class_windows_val)
        metrics_windows['recall'] = recall_score(class_true_win, class_windows_val)

        metrics[inner_steps]['time_instance'] = metrics_time_instance
        metrics[inner_steps]['windows'] = metrics_windows

        log_analysis[inner_steps] = {
            'err_val': err_val,
            'err_val_win': err_val_win,
            'err_val_norm2': err_val_norm2,
            'err_val_norm2_win': err_val_norm2_win,
            'err_test': err_test,
            'err_test_win': err_test_win,
            'err_test_norm2': err_test_norm2,
            'err_test_norm2_win': err_test_norm2_win,
            'class_true': class_true,
            'class_true_win': class_true_win,
            'class_point_val_feat': class_point_val_feat,
            'class_point_val': class_point_val,
            'class_windows_val_feat': class_windows_val_feat,
            'class_windows_val': class_windows_val,
            'th_from_val': th_from_val,
            'th_from_val_win': th_from_val_norm2_win,
            'list_f1_score': list_f1_single,
            'list_f1PA_score': list_f1PA_single,
            'list_f1COM_score': list_f1COM_single,
            'list_f1AFF_score': list_f1AFF_single,
            'list_f1_score_win': list_f1_win,
            'list_f1PA_score_win': list_f1PA_win,
            'list_f1COM_score_win': list_f1COM_win,
            'list_f1AFF_score_win': list_f1COM_win,
            'psnr_test_per_feat': psnr_test_per_feat,
            'psnr_train_per_feat': psnr_train_per_feat,
            'psnr_val_per_feat': psnr_val_per_feat,
        }

        print(f'class_windows_val_feat {class_windows_val_feat.shape}')
        print(f'err_test_win {err_test_win.shape}')
        print(f'class_true {class_true.shape}')
        if args.save_figures_feat:
            log_analysis[inner_steps]['figures'] = {'rec': inner_log['rec_figures'],
                                                    'mask': {
                                                        'err_test': get_mask(err_test, log_scale=True,
                                                                             title='Error test per time instance - logscaled'),
                                                        'err_test_win': get_mask(err_test_win, log_scale=True,
                                                                                 title='Error test per windows - logscaled'),
                                                        'class_point_val_feat': get_mask(
                                                            class_point_val_feat.astype(float), log_scale=False,
                                                            title=f'Class per time instance - threshold {args.quantile_th} val'),
                                                        'class_windows_val_feat': get_mask(
                                                            class_windows_val_feat.astype(float), log_scale=False,
                                                            title=f'Class per windows - threshold {args.quantile_th} val'),
                                                        'class_true': get_mask(class_true[:, None].astype(float),
                                                                               log_scale=False,
                                                                               title='Class per time instance - ground truth'),
                                                        'class_true_win': get_mask(
                                                            class_true_win[:, None].astype(float), log_scale=False,
                                                            title='Class per windows - ground truth')}
                                                    }

        if args.plot_metrics:
            plot_roc(fpr, tpr,
                     savename=f'single_ROC_curve_inner_steps_{inner_steps}.png',
                     title=f'ROC single time instance inner_steps {inner_steps} AUC {auc_single:0.4f}',
                     dirpath=dirpath_analysis)

            plot_roc(fpr_win, tpr_win,
                     savename=f'win_ROC_curve_inner_steps_{inner_steps}.png',
                     title=f'ROC windows inner_steps {inner_steps} AUC {auc_win:0.4f}',
                     dirpath=dirpath_analysis)

            plot_err(err_test_norm2, class_true,
                     title=f'Error norm2',
                     savename=f'single_err_inner_steps_{inner_steps}.png',
                     dirpath=dirpath_analysis)

            plot_err(err_test_norm2_win, class_true_win,
                     title=f'Error norm2 windows',
                     savename=f'win_err_inner_steps_{inner_steps}.png',
                     dirpath=dirpath_analysis)

            plot_err_distribution(err_test_norm2, class_true,
                                  title=f'Error norm2 distribution',
                                  savename=f'single_err_distr_inner_steps_{inner_steps}.png',
                                  dirpath=dirpath_analysis)

            plot_err_distribution(err_test_norm2_win, class_true_win,
                                  title=f'Error norm2 windows distribution',
                                  savename=f'win_err_distr_inner_steps_{inner_steps}.png',
                                  dirpath=dirpath_analysis)

            plot_f1_th(list_f1_single,
                       ths,
                       th_from_val_norm2,
                       title=f'F1 score single vs threshold',
                       savename='single_f1_th.png',
                       dirpath=dirpath_analysis)

            plot_f1_th(list_f1_win,
                       ths_win,
                       th_from_val_norm2_win,
                       title=f'F1 score win vs threshold',
                       savename='win_f1_th.png',
                       dirpath=dirpath_analysis)

    # print a summary of the computed metrics
    for k, v in metrics.items():
        print(f'inner_steps {k}')
        for k1, v1 in v.items():
            print(f'\t {k1}')
            for k2, v2 in v1.items():
                print(f'\t\t{k2:>20s} = {v2:.5f}')

    # save the metrics, analysis and metadata
    with (dirpath_analysis / 'metrics.json').open('w') as f:
        json.dump(metrics, f, indent=4)

    with (dirpath_analysis / 'log_analysis.pkl').open('wb') as f:
        pickle.dump(log_analysis, f)

    with (dirpath_analysis / 'metadata.pkl').open('wb') as f:
        pickle.dump(metadata, f)

    return metrics


def main(args):
    """This main function serves as the entry point for the program. It takes the command line arguments as input and returns a dictionary of metrics obtained from the log data.
    The function performs the following steps:
    1) Load the model configuration using the load_model_config function. This includes retrieving the model state dictionary, training arguments, and other relevant information.
    2) Run the training loop using the run_loop function. This function executes the training process and logs the relevant information into the log_dict.
    3) Analyze the log data and compute metrics using the analyse_log function. This function processes the log data, calculates various metrics, and returns them as a dictionary.
    Return the metrics obtained from the log analysis.

    Args:
        args: The command line arguments parsed using argparse.

    Returns:
        metrics: A dictionary containing the analysis metrics obtained from the log data.
    """

    # Load the model configuration
    model_state_dict, args_train, wandb_dir, dirpath_analysis, model_epoch = load_model_config(args)

    # Run the training loop
    log_dict = run_loop(args, model_state_dict, args_train, dirpath_analysis)

    # Analyze the log data and compute metrics
    metrics = analyse_log(args, args_train, log_dict, dirpath_analysis, model_epoch)

    return metrics


if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = main(args)
