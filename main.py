import argparse
import helpers
import os
import pprint
import yaml
import torch
import wandb
from coinpp.training import Trainer
from pathlib import Path


def add_arguments(parser):
    """Model, training and wandb arguments. Note that we use integers for boolean
    arguments due to compatibility issues with wandb.
    """
    # Model arguments
    parser.add_argument(
        "--num_layers",
        help="Number of layers in base network.",
        type=int,
        default=10
    )

    parser.add_argument(
        "--dim_hidden",
        help="Dimension of hidden layers of base network.",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--multi_branch",
        type=int,
        default=0,
        help="Whether to use multi-branch architecture."
    )

    parser.add_argument(
        '--modulate_trunk',
        default=1,
        type=int,
        help='Whether to modulate the trunk.'
    )

    parser.add_argument(
        "--num_layers_trunk",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num_layers_branch",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dim_hidden_trunk",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--dim_hidden_branch",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--residual",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--w0",
        help="w0 parameter from SIREN.",
        type=float,
        default=500.0
    )

    parser.add_argument(
        "--s0",
        help="w0 parameter from WIRE.",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--modulate_scale",
        help="Whether to modulate scale.",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--VanillaMAML",
        help="Use the VanillaMAML meta-learning algorithm.",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--modulate_shift",
        help="Whether to modulate shift.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--use_latent",
        help="Whether to use latent vector.",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--latent_dim",
        help="Dimension of the latent vector mapped to modulations. If set to -1, will not use latent vector.",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--modulation_net_dim_hidden",
        help="Dimension of hidden layers in modulation network.",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--modulation_net_num_layers",
        help="Number of layers in modulation network. 1 corresponds to a linear layer.",
        type=int,
        default=1,
    )

    # Training arguments
    parser.add_argument(
        "--seed",
        help="Random seed. If set to -1, seed is chosen at random.",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--outer_lr",
        help="Learning rate for the outer loop.",
        type=float,
        default=1e-6
    )

    parser.add_argument(
        "--inner_lr",
        help="Learning rate for the inner loop.",
        type=float,
        default=1e-2
    )

    parser.add_argument(
        "--inner_steps",
        help="Number of inner loop steps.",
        type=int,
        default=3
    )

    parser.add_argument(
        '--parallel',
        help='Whether to use parallel model, i.e. multiple unviariate models on the same input, one for each output feature.',
        default=0,
        type=int,
    )

    parser.add_argument(
        '--parallel_matrix',
        help='Whether to use one model with parallel linear weights matrices, simulating multiple unviariate models on the same input, one for each output feature.',
        default=0,
        type=int,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30000
    )

    parser.add_argument(
        "--last_layer_modulated",
        type=int,
        default=0
    )

    parser.add_argument(
        "--train_dataset",
        default="smd",
        choices=(
            "librispeech",
            "smd",
            'swat',
            "ar_time_series",
            'wadi'
        ),
    )

    parser.add_argument(
        "--test_dataset",
        default="smd",
        choices=(
            "librispeech",
            "smd",
            'swat',
            "ar_time_series",
            'wadi'
        ),
    )

    parser.add_argument(
        "--normalization_kind",
        default="plus1div2",
        choices=(
            ("minmax", "standard", "plus1div2", None)
        ),
    )

    parser.add_argument(
        '--skip_first_n_samples',
        default=0,
        type=int,
        help='Number of samples to skip at the beginning of the time series. '
             'Suggested to be set to 21600 for WADI and SWAT, 0 for the others.'
    )

    parser.add_argument(
        '--downsampling_factor',
        default=1,
        type=int,
        help='Downsample the time series by this factor.'
    )

    parser.add_argument(
        "--train_proportion",
        help="Proportion of training_samples in the train+val split",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--use_bias",
        help="Whether to use bias in the base network.",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--num_workers",
        help="Number of workers for dataloader.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient_checkpointing",
        help="Whether to use gradient checkpointing.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num_validation_points",
        help="Number of validation data points to use. If -1, will use all available points.",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--validate_every",
        help="Run validation every {validate_every} iterations.",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--log_every",
        help="Print log every {logs_every} steps.",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--validation_inner_steps",
        help="List of inner steps to use for validation.",
        nargs="+",
        type=int,
        default=[3],
    )

    parser.add_argument(
        "--validation_inner_lr",
        help="inner lr to use for validation.",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        '--full_anomaly_detection',
        default=0,
        type=int,
        help='Whether to consider input data as anomaly detection task or not. '
         'If 1, will plot and analyse the anomaly detection results. '
    )

    parser.add_argument(
        "--patch_shape",
        help="Shape of patches to use during training. If set to [-1] will not apply patching.",
        nargs="+",
        type=int,
        default=[-1],
    )

    parser.add_argument(
        "--window_length",
        help="Valid only for time-series data. Length of time-series windows to consider as samples. "
             "If set to [-1], do not split the original time-series.",
        nargs="+",
        type=int,
        default=[10],
    )

    # parser.add_argument(
    #     "--skip_first_n_samples",
    #     help="Valid only for WADI dataset. Ignore the first samples of train data (to be deemed as warm up points). "
    #          "If set to [-1], do not split the original time-series.",
    #     nargs="+",
    #     type=int,
    #     default=21160,
    # )

    parser.add_argument(
        "--entity",
        help="Valid only for SMD time-series dataset. Select the entity to consider.",
        type=str,
        default='machine-1-1',
    )

    parser.add_argument(
        "--selected_features",
        help="Valid only for SMD time-series dataset. Select the list of channels of the entity to consider"
             "If set to [-1], consider all the channels",
        nargs="+",
        type=int,
        default=[-1],
    )

    parser.add_argument(
        "--nsample",
        help="Valid only for AR_TS time-series dataset. Number of samples to generate.",
        type=int,
        default=1000,
    )

    parser.add_argument(
        '--ar_param_list',
        help='Valid only for AR_TS time-series dataset. List of AR parameters to use for the AR process.',
        nargs='+',
        type=float,
        default=[0.5, 0.3, 0.1],
    )

    parser.add_argument(
        "--subsample_num_points",
        help="Number of coordinate points to subsample during training. If -1, uses full datapoint/patch.",
        type=int,
        default=-1,
    )

    # Wandb arguments
    parser.add_argument(
        "--use_wandb",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="debug",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--wandb_job_type",
        help="Wandb job type. This is useful for grouping runs together.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--wandb_tags",
        nargs='+',
        type=str,
        default=[],
    )

    parser.add_argument(
        "--wandb_mode",
        help="Wandb mode type. online, offline or dry-run (see wandb documentation).",
        type=str,
        default='offline',
    )

    parser.add_argument(
        "--wandb_watch",
        help="Wandb watch all (see wandb doc of wandb.watch).",
        type=int,
        default=0,
    )

    parser.add_argument(
        '--save_models',
        default='best_val',
        type=str,
        help='save the model with the best validation loss', choices=('best_val', 'all')
    )
    parser.add_argument(
        '--plot_wandb',
        default=0,
        type=int
    )


def main(args):
    if args.use_wandb:
        # Initialize wandb experiment
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            job_type=args.wandb_job_type,
            config=args,
            mode=args.wandb_mode,
            tags=args.wandb_tags
        )

        # Save ENV variables
        with (Path(wandb.run.dir) / "env.txt").open("wt") as f:
            pprint.pprint(dict(os.environ), f)

        # Save config_args variables
        with (Path(wandb.run.dir) / "config_args.yaml").open("w") as f:
            yaml.dump(args.__dict__, f)

        # Define path where model will be saved
        model_path = Path(wandb.run.dir) / "model_epoch{:07d}.pt"
        config_args_path = Path(wandb.run.dir) / "config_args.yaml"

    else:
        model_path = Path(".") / "model.pt"
        config_args_path = Path('.') / "config_args.yaml"

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optionally set random seed
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Build datasets, converters and model
    train_dataset, test_dataset, converter = helpers.get_datasets_and_converter(args)
    model = helpers.get_model(args)

    print(model)
    print(args)

    # Optionally save model
    if args.use_wandb:
        # torch.save({"args": args, "state_dict": model.state_dict()}, model_path)
        wandb.save(str(model_path.absolute()), base_path=wandb.run.dir, policy="live")
        wandb.save(str(config_args_path.absolute()), base_path=wandb.run.dir, policy="live")
        if args.wandb_watch:
            wandb.watch(model, log='all',  log_freq= 1)

    # Initialize trainer and start training
    trainer = Trainer(
        func_rep=model,
        converter=converter,
        args=args,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_path=model_path,
    )

    for epoch in range(1, args.num_epochs + 1):
        if epoch % 50 == 0:
            print(f"\nEpoch {epoch}:")
        trainer.train_epoch(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
