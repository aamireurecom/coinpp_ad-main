import pathlib
MAX_LOGS_LOAD = 2
F1_SCORE_TH_NUM_DEFAULT = 201

# WANDB_PATH = pathlib.Path('wandb/runs_server')
WANDB_PATH = pathlib.Path('wandb')
# SAVE_ALL_DUMP = pathlib.Path('all_dump.pkl')
FILENAME_LOCAL_DUMP = 'local_dump.pkl'
FILEPATH_GLOBAL_DUMP = pathlib.Path('global_dump.pkl')

METRICS_TIME_INSTANCE = ['auc_score', 'f1_score', 'best_f1_score', 'best_f1PA_score', 'best_f1COM_score',
                         'best_f1AFF_score', 'best_precision', 'best_recall', 'train_psnr', 'val_psnr', 'test_psnr']
METRICS_WINDOWS = ['auc_score', 'f1_score', 'best_f1_score', 'best_f1PA_score', 'best_f1COM_score', 'best_f1AFF_score', 'best_precision', 'best_recall']

DICT_TABLE_INIT = {'time_instance':
                       {'run': [], 'auc_score': [], 'f1_score': [],
                                     'best_f1_score': [], 'best_f1PA_score': [], 'best_f1COM_score': [], 'best_f1AFF_score': [],
                        'best_precision': [], 'best_recall': [], f'train_psnr': [], f'val_psnr': [], f'test_psnr': [],
                                     # 'model_epoch': [],
                        'entity': [], 'wandb_job_type': [], 'train_dataset': [],
                       },
                   'windows':
                       {'run': [], 'auc_score': [], 'f1_score': [],
                        'best_f1_score': [], 'best_f1PA_score': [], 'best_f1COM_score': [], 'best_f1AFF_score': [],
                        'best_precision': [], 'best_recall': [],
                        'entity': [], 'wandb_job_type': [], 'train_dataset': [],}
                   }