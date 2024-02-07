import coinpp.conversion as conversion
import coinpp.models as models
import data.traffic as traffic
import data.swat as swat
import data.wadi as wadi
import data.ar_ts as ar_ts
import data.audio as audio

import torchvision
import yaml
from pathlib import Path


def get_dataset_root(dataset_name: str):
    """Returns path to data based on dataset_paths.yaml file."""
    with open(r"data/dataset_paths.yaml") as f:
        dataset_paths = yaml.safe_load(f)

    return Path(dataset_paths[dataset_name])


def dataset_name_to_dims(dataset_name, channels=(0,)):
    """Returns appropriate dim_in and dim_out for dataset."""
    if dataset_name == "librispeech":
        dim_in = 1
        dim_out = 1
    elif dataset_name == 'smd':
        dim_in = 1
        if channels[0] == -1:
            dim_out = 38
        else:
            dim_out = len(channels)
    elif dataset_name == 'swat':
        dim_in = 1
        if channels[0] == -1:
            dim_out = 51
        else:
            dim_out = len(channels)
    elif dataset_name == 'wadi':
        dim_in = 1
        if channels[0] == -1:
            dim_out = 127
        else:
            dim_out = len(channels)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dim_in, dim_out


def get_datasets_and_converter(args, force_no_random_crop=False):
    """Returns train and test datasets as well as appropriate data converters.

    Args:
        args: Arguments parsed from input.
        force_no_random_crop (bool): If True, forces datasets to not use random
            crops (which is the default for the training set when using
            patching). This is useful after the model is trained when we store
            modulations.
    """
    # Extract input and output dimensions of function rep
    dim_in, dim_out = dataset_name_to_dims(args.train_dataset,channels=args.selected_features)

    # When using patching, perform random crops equal to patch size on training
    # dataset
    use_patching = hasattr(args, "patch_shape") and args.patch_shape != [-1]
    if use_patching:
        if dim_in == 2:
            random_crop = torchvision.transforms.RandomCrop(args.patch_shape)

    # TODO: inconsistent use of args.train_dataset and args.test_dataset inheredited from coinpp repo.
    # TODO: Implement choice of different dataset for train and test

    if "librispeech" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter("audio")

        # We use first 3 seconds of each audio sample
        if args.train_dataset == "librispeech":
            train_dataset = audio.LIBRISPEECH(
                root=get_dataset_root("librispeech"),
                url="train-clean-100",
                patch_shape=args.patch_shape[0]
                if (use_patching and not force_no_random_crop)
                else -1,
                num_secs=3,
                download=True,
            )
        if args.test_dataset == "librispeech":
            test_dataset = audio.LIBRISPEECH(
                root=get_dataset_root("librispeech"),
                url="test-clean",
                num_secs=3,
                download=True,
            )

    elif "smd" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter('time-series')

        if args.train_dataset == "smd":
            train_dataset = traffic.SMD(
                root=get_dataset_root("smd"),
                split='train',
                patch_shape=args.patch_shape[0]
                if (use_patching and not force_no_random_crop)
                else -1,
                entity=args.entity,
                window_length=args.window_length[0],
                selected_features=args.selected_features
            )
        if args.test_dataset == "smd":
            test_dataset = {
                'train': traffic.SMD(
                    root=get_dataset_root("smd"),
                    split='train',
                    entity=args.entity,
                    selected_features=args.selected_features,
                    window_length=args.window_length[0],
                    train_proportion=args.train_proportion,
                    # patch_shape=args.patch_shape[0]
                ),
                'val': traffic.SMD(
                    root=get_dataset_root("smd"),
                    split='val',
                    entity=args.entity,
                    selected_features=args.selected_features,
                    window_length = args.window_length[0],
                    train_proportion=args.train_proportion,
                    # patch_shape=args.patch_shape[0]
                ),
                'test': traffic.SMD(
                    root=get_dataset_root("smd"),
                    split='test',
                    entity=args.entity,
                    selected_features=args.selected_features,
                    window_length=args.window_length[0],
                    train_proportion=args.train_proportion,
                    # patch_shape=args.patch_shape[0]
            )
                }
    elif "swat" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter('time-series')

        if args.train_dataset == "swat":
            train_dataset = swat.SWAT(
                root=get_dataset_root("swat"),
                split='train',
                patch_shape=args.patch_shape[0]
                if (use_patching and not force_no_random_crop)
                else -1,
                window_length=args.window_length[0],
                selected_features=args.selected_features,
                skip_first_n_samples=args.skip_first_n_samples,
                downsampling_factor = args.downsampling_factor

            )
        if args.test_dataset == "swat":
            test_dataset = {
                'train': swat.SWAT(
                    root=get_dataset_root("swat"),
                    split='train',
                    selected_features=args.selected_features,
                    window_length=args.window_length[0],
                    train_proportion=args.train_proportion,
                    skip_first_n_samples=args.skip_first_n_samples,
                    downsampling_factor=args.downsampling_factor
                    # patch_shape=args.patch_shape[0]
                ),
                'val': swat.SWAT(
                    root=get_dataset_root("swat"),
                    split='val',
                    selected_features=args.selected_features,
                    window_length=args.window_length[0],
                    train_proportion=args.train_proportion,
                    skip_first_n_samples=args.skip_first_n_samples,
                    downsampling_factor=args.downsampling_factor

                    # patch_shape=args.patch_shape[0]
                ),
                'test': swat.SWAT(
                    root=get_dataset_root("swat"),
                    split='test',
                    selected_features=args.selected_features,
                    window_length=args.window_length[0],
                    train_proportion=args.train_proportion,
                    skip_first_n_samples=args.skip_first_n_samples,
                    downsampling_factor=args.downsampling_factor

                    # patch_shape=args.patch_shape[0]
                )
            }

    elif "wadi" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter('time-series')

        # We use machine-1-1 source dataset
        # TODO: Account for the choice of different datasets

        if args.train_dataset == "wadi":
            train_dataset = wadi.WADI(
                root=get_dataset_root("wadi"),
                split='train',
                patch_shape=args.patch_shape[0]
                if (use_patching and not force_no_random_crop)
                else -1,
                window_length=args.window_length[0],
                selected_features=args.selected_features,
                skip_first_n_samples=args.skip_first_n_samples,
                downsampling_factor=args.downsampling_factor
            )
        if args.test_dataset == "wadi":
            test_dataset = {
                'train': wadi.WADI(
                    root=get_dataset_root("wadi"),
                    split='train',
                    selected_features=args.selected_features,
                    window_length=args.window_length[0],
                    train_proportion=args.train_proportion,
                    skip_first_n_samples=args.skip_first_n_samples,
                    downsampling_factor = args.downsampling_factor
            # patch_shape=args.patch_shape[0]
                ),
                'val': wadi.WADI(
                    root=get_dataset_root("wadi"),
                    split='val',
                    selected_features=args.selected_features,
                    window_length=args.window_length[0],
                    train_proportion=args.train_proportion,
                    skip_first_n_samples=args.skip_first_n_samples,
                    downsampling_factor=args.downsampling_factor

                    # patch_shape=args.patch_shape[0]
                ),
                'test': wadi.WADI(
                    root=get_dataset_root("wadi"),
                    split='test',
                    selected_features=args.selected_features,
                    window_length=args.window_length[0],
                    train_proportion=args.train_proportion,
                    skip_first_n_samples=args.skip_first_n_samples,
                    downsampling_factor=args.downsampling_factor
                    # patch_shape=args.patch_shape[0]
                )
            }
    elif "ar_ts" in (args.train_dataset, args.test_dataset):
        converter = conversion.Converter('time-series')

        if args.train_dataset == "ar_ts":
            train_dataset = ar_ts.ARTimeSeries(
                split='train',
                window_length=args.window_length[0],
                selected_features=args.selected_features
            )
        if args.test_dataset == "ar_ts":
            test_dataset = {
                'train': ar_ts.ARTimeSeries(
                    split='train',
                    selected_features=args.selected_features,
                    window_length=args.window_length[0],
                    train_proportion=args.train_proportion,
                    ar_param_list=args.ar_param_list,
                    nsample=args.nsample,
                    # patch_shape=args.patch_shape[0]
                ),
                'val': ar_ts.ARTimeSeries(
                    split='val',
                    selected_features=args.selected_features,
                    window_length=args.window_length[0],
                    train_proportion=args.train_proportion,
                    ar_param_list=args.ar_param_list,
                    nsample=args.nsample,
                    # patch_shape=args.patch_shape[0]
                ),
                'test': ar_ts.ARTimeSeries(
                    split='test',
                    selected_features=args.selected_features,
                    window_length=args.window_length[0],
                    train_proportion=args.train_proportion,
                    ar_param_list=args.ar_param_list,
                    nsample=args.nsample,
                    # patch_shape=args.patch_shape[0]
                )
            }
    else:
        raise NotImplementedError(f"Dataset {args.train_dataset} {args.test_dataset} not implemented")
    return train_dataset, test_dataset, converter


def get_model(args):
    dim_in, dim_out = dataset_name_to_dims(args.train_dataset, channels=args.selected_features)
    if args.VanillaMAML:
        return models.Siren(
            dim_in=dim_in,
            dim_hidden=args.dim_hidden,
            dim_out=dim_out,
            num_layers=args.num_layers,
            w0=args.w0,
            w0_initial=args.w0,
            use_bias=args.use_bias,
        ).to(args.device)

    else:
        if 'parallel' in args and args.parallel:
            print('*' * 80)
            print('Loading residual model')
            return models.ModulatedSirenParallel(
                dim_in=dim_in,
                dim_hidden=args.dim_hidden,
                dim_out=dim_out,
                num_layers=args.num_layers,
                w0=args.w0,
                w0_initial=args.w0,
                use_bias=args.use_bias,
                modulate_scale=args.modulate_scale,
                modulate_shift=args.modulate_shift,
                use_latent=args.use_latent,
                latent_dim=args.latent_dim,
                modulation_net_dim_hidden=args.modulation_net_dim_hidden,
                modulation_net_num_layers=args.modulation_net_num_layers,
                last_layer_modulated=args.last_layer_modulated,
            ).to(args.device)
        elif 'parallel_matrix' in args and args.parallel_matrix:
            print('*' * 80)
            print('Loading residual model')
            return models.ModulatedSirenParallelMatrix(
                dim_in=dim_in,
                dim_hidden=args.dim_hidden,
                dim_out=dim_out,
                num_layers=args.num_layers,
                w0=args.w0,
                w0_initial=args.w0,
                use_bias=args.use_bias,
                modulate_scale=args.modulate_scale,
                modulate_shift=args.modulate_shift,
                use_latent=args.use_latent,
                latent_dim=args.latent_dim,
                modulation_net_dim_hidden=args.modulation_net_dim_hidden,
                modulation_net_num_layers=args.modulation_net_num_layers,
                last_layer_modulated=args.last_layer_modulated,
            ).to(args.device)

        elif 'residual' in args and args.residual:
            print('*' * 80)
            print('Loading residual model')
            return models.ModulatedSirenResidual(
                dim_in=dim_in,
                dim_hidden=args.dim_hidden,
                dim_out=dim_out,
                num_layers=args.num_layers,
                w0=args.w0,
                w0_initial=args.w0,
                use_bias=args.use_bias,
                modulate_scale=args.modulate_scale,
                modulate_shift=args.modulate_shift,
                use_latent=args.use_latent,
                latent_dim=args.latent_dim,
                modulation_net_dim_hidden=args.modulation_net_dim_hidden,
                modulation_net_num_layers=args.modulation_net_num_layers,
                last_layer_modulated=args.last_layer_modulated,
            ).to(args.device)
        if 'multi_branch' in args and args.multi_branch:
            print('*' * 80)
            print('Loading multi-branch model')
            return models.ModulatedSirenMultiBranch(
                dim_in=dim_in,
                dim_hidden_trunk=args.dim_hidden_trunk,
                dim_hidden_branch=args.dim_hidden_branch,
                dim_out=dim_out,
                num_layers_trunk=args.num_layers_trunk,
                num_layers_branch=args.num_layers_branch,
                w0=args.w0,
                w0_initial=args.w0,
                use_bias=args.use_bias,
                modulate_scale=args.modulate_scale,
                modulate_shift=args.modulate_shift,
                use_latent=args.use_latent,
                latent_dim=args.latent_dim,
                modulation_net_dim_hidden=args.modulation_net_dim_hidden,
                modulation_net_num_layers=args.modulation_net_num_layers,
                last_layer_modulated=args.last_layer_modulated,
                modulate_trunk=args.modulate_trunk,
            ).to(args.device)

        else:
            return models.ModulatedSiren(
                dim_in=dim_in,
                dim_hidden=args.dim_hidden,
                dim_out=dim_out,
                num_layers=args.num_layers,
                w0=args.w0,
                w0_initial=args.w0,
                use_bias=args.use_bias,
                modulate_scale=args.modulate_scale,
                modulate_shift=args.modulate_shift,
                use_latent=args.use_latent,
                latent_dim=args.latent_dim,
                modulation_net_dim_hidden=args.modulation_net_dim_hidden,
                modulation_net_num_layers=args.modulation_net_num_layers,
                last_layer_modulated=args.last_layer_modulated,
            ).to(args.device)


def get_model_single(args):
    dim_in, dim_out = dataset_name_to_dims(args.train_dataset, channels=args.selected_features)
    if args.multi_branch:
        return models.SirenMultiBranch(
            dim_in=dim_in,
            dim_hidden_trunk=args.dim_hidden_trunk,
            dim_hidden_branch=args.dim_hidden_branch,
            dim_out=dim_out,
            num_layers_trunk=args.num_layers_trunk,
            num_layers_branch=args.num_layers_branch,
            w0=args.w0,
            w0_initial=args.w0,
        ).to(args.device)

    else:
        return models.Siren(
            dim_in=dim_in,
            dim_hidden=args.dim_hidden,
            dim_out=dim_out,
            num_layers=args.num_layers,
            w0=args.w0,
            w0_initial=args.w0,
        ).to(args.device)
