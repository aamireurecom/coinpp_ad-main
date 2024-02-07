import coinpp.conversion as conversion
import coinpp.losses as losses
import coinpp.metalearning_single as metalearning
import torch
import wandb
import matplotlib.pyplot as plt

class TrainerSingle:
    def __init__(
        self,
        func_rep,
        converter,
        args,
        train_dataset,
        test_dataset,
        patcher=None,
        model_path="",
        device='cuda',
    ):
        # TODO: Rewrite docs
        """Module to handle meta-learning of COIN++ model.

        Args:
            func_rep (models.ModulatedSiren):
            converter (conversion.Converter):
            args: Training arguments (see main.py).
            train_dataset:
            test_dataset:
            patcher: If not None, patcher that is used to create random patches during
                training and to partition data into patches during validation.
            model_path: If not empty, wandb path where best (validation) model
                will be saved.
        """
        self.func_rep = func_rep
        self.converter = converter
        self.args = args
        self.patcher = patcher

        self.optimizer = torch.optim.Adam(
            self.func_rep.parameters(), lr=args.start_outer_lr
        )

        self.outer_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=args.factor_reduce_outer_lr_plateau,
                                                                    patience=args.patience_val_steps_reduce_outer_lr_plateau,
                                                                    threshold=args.threshold_reduce_outer_lr_plateau,
                                                                    threshold_mode='rel', cooldown=0,
                                                                    min_lr=args.end_outer_lr, eps=1e-08,
                                                                    verbose=True)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset #['train']
        self._process_datasets()
        self.stft_loss = losses.MultiResolutionSTFTLoss(fft_sizes=args.stft_fft_sizes,
                                                        hop_sizes=args.stft_hop_sizes,
                                                        win_lengths=args.stft_win_lengths
                                                        )

        self.model_path = model_path
        self.step = 0
        self.best_val_psnr = 0.0

    def _process_datasets(self):
        """Create dataloaders for datasets based on self.args."""
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.num_workers > 0,
        )

        # If we are using patching, require data loader to have a batch size of 1,
        # since we can potentially have different sized outputs which cannot be batched
        # NOTE: TRAIN_DATASET == TEST_DATASET for TrainerSingle
        self.test_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=False,
            batch_size=1 if self.patcher else self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def train_epoch(self):
        """Train model for a single epoch."""

        for data in self.train_dataloader:

            data = data.to(self.args.device)
            coordinates, features = self.converter.to_coordinates_and_features(data)

            # Optionally subsample points
            if self.args.subsample_num_points != -1:
                # Coordinates have shape (batch_size, *, coordinate_dim)
                # Features have shape (batch_size, *, feature_dim)
                # Flatten both along spatial dimension and randomly select points
                coordinates = coordinates.reshape(
                    coordinates.shape[0], -1, coordinates.shape[-1]
                )
                features = features.reshape(features.shape[0], -1, features.shape[-1])
                # Compute random indices (no good pytorch function to do this,
                # so do it this slightly hacky way)
                permutation = torch.randperm(coordinates.shape[1])
                idx = permutation[: self.args.subsample_num_points]
                coordinates = coordinates[:, idx, :]
                features = features[:, idx, :]

            outputs = metalearning.outer_step(
                self.func_rep,
                coordinates,
                features,
                loss_fn_stft=self.stft_loss,
                sftf_loss_weight=self.args.sftf_loss_weight,
                # inner_steps=self.args.inner_steps,
                # inner_lr=self.args.inner_lr,
                is_train=True,
                return_reconstructions=False,
                # gradient_checkpointing=self.args.gradient_checkpointing,
            )

            # Update parameters of base network
            self.optimizer.zero_grad()
            outputs["loss"].backward(create_graph=False)
            self.optimizer.step()

            log_dict = {
                "loss": outputs["loss"].item(),
                "loss_mse": outputs["loss_mse"].item(),
                "loss_stft_l1": outputs["loss_stft_l1"].item(),
                "loss_stft_log": outputs["loss_stft_log"].item(),
                "psnr": outputs["psnr"].item(),
                'outer_lr': [group['lr'] for group in self.optimizer.param_groups][0]
            }

            if self.args.use_wandb:
                wandb.log(log_dict, step=self.step)

            self.step += 1

            if (self.step % 100) == 0:
                print(
                    f'Step {self.step}, '
                    f'Loss {log_dict["loss"]:.3f}, '
                    f"Loss_MSE {log_dict['loss_mse']:.3f}, "
                    f"Loss_STFT_l1 {log_dict['loss_stft_l1']:.3f}, "
                    f"Loss_STFT_log {log_dict['loss_stft_log']:.3f}, "
                    f'PSNR {log_dict["psnr"]:.3f}'
                )

            if self.step % self.args.validate_every == 0:
                val_psnr = self.validation()
                self.outer_lr_scheduler.step(val_psnr)

    def validation(self):
        """Run trained model on validation dataset."""
        print(f"\nValidation, Step {self.step}:")
        old_coordinates = None

        # If num_validation_points is -1, validate on entire validation dataset,
        # otherwise validate on a subsample of points
        full_validation = self.args.num_validation_points == -1
        num_validation_batches = self.args.num_validation_points // self.args.batch_size

        # Initialize validation logging dict
        log_dict = {}

        # Evaluate model for different numbers of inner loop steps
        for inner_steps in self.args.validation_inner_steps:
            log_dict[f"val_psnr_{inner_steps}_steps"] = 0.0
            log_dict[f"val_loss_{inner_steps}_steps"] = 0.0
            log_dict[f"val_loss_mse_{inner_steps}_steps"] = 0.0
            log_dict[f"val_loss_stft_l1_{inner_steps}_steps"] = 0.0
            log_dict[f"val_loss_stft_log_{inner_steps}_steps"] = 0.0

            # Fit modulations for each validation datapoint
            for i, data in enumerate(self.test_dataloader):
                data = data.to(self.args.device)
                if self.patcher:
                    # If using patching, test data will have a batch size of 1.
                    # Remove batch dimension and instead convert data into
                    # patches, with patch dimension acting as batch size
                    patches, spatial_shape = self.patcher.patch(data[0])
                    coordinates, features = self.converter.to_coordinates_and_features(
                        patches
                    )

                    # As num_patches may be much larger than args.batch_size,
                    # split the fitting of patches into batch_size chunks to
                    # reduce memory
                    outputs = metalearning.outer_step_chunked(
                        self.func_rep,
                        coordinates,
                        features,
                        loss_fn_stft=self.stft_loss,
                        sftf_loss_weight=self.args.sftf_loss_weight,
                        # inner_steps=inner_steps,
                        # inner_lr=self.args.inner_lr,
                        chunk_size=self.args.batch_size,
                        # gradient_checkpointing=self.args.gradient_checkpointing,
                    )

                    # Shape (num_patches, *patch_shape, feature_dim)
                    patch_features = outputs["reconstructions"]

                    # When using patches, we cannot directly use psnr and loss
                    # output by outer step, since these are calculated on the
                    # padded patches. Therefore we need to reconstruct the data
                    # in its original unpadded form and manually calculate mse
                    # and psnr
                    # Shape (num_patches, *patch_shape, feature_dim) ->
                    # (num_patches, feature_dim, *patch_shape)
                    patch_data = conversion.features2data(patch_features, batched=True)
                    # Shape (feature_dim, *spatial_shape)
                    data_recon = self.patcher.unpatch(patch_data, spatial_shape)
                    # Calculate MSE and PSNR values and log them
                    mse = losses.mse_fn(data_recon, data[0])
                    psnr = losses.mse2psnr(mse)
                    # print(data_recon.shape)
                    stft_l1, stft_log = self.stft_loss(data_recon, data[0])
                    loss = mse + self.args.sftf_loss_weight * (stft_l1 + stft_log)

                    log_dict[f"val_psnr_{inner_steps}_steps"] += psnr.item()
                    log_dict[f"val_loss_{inner_steps}_steps"] += loss.item()
                    log_dict[f"val_loss_mse_{inner_steps}_steps"] += mse.item()
                    log_dict[f"val_loss_stft_l1_{inner_steps}_steps"] += stft_l1.item()
                    log_dict[f"val_loss_stft_log_{inner_steps}_steps"] += stft_log.item()
                else:
                    old_coordinates = self.converter.coordinates
                    self.converter.coordinates = None
                    coordinates, features = self.converter.to_coordinates_and_features(
                        data
                    )

                    outputs = metalearning.outer_step(
                        self.func_rep,
                        coordinates,
                        features,
                        loss_fn_stft=self.stft_loss,
                        sftf_loss_weight=self.args.sftf_loss_weight,
                        # inner_steps=inner_steps,
                        # inner_lr=self.args.inner_lr,
                        is_train=False,
                        return_reconstructions=True,
                        # gradient_checkpointing=self.args.gradient_checkpointing,
                    )

                    log_dict[f"val_psnr_{inner_steps}_steps"] += outputs["psnr"].item()
                    log_dict[f"val_loss_{inner_steps}_steps"] += outputs["loss"].item()
                    log_dict[f"val_loss_mse_{inner_steps}_steps"] += outputs["loss_mse"].item()
                    log_dict[f"val_loss_stft_l1_{inner_steps}_steps"] += outputs["loss_stft_l1"].item()
                    log_dict[f"val_loss_stft_log_{inner_steps}_steps"] += outputs["loss_stft_log"].item()

                if not full_validation and i >= num_validation_batches - 1:
                    break

            # Calculate average PSNR and loss by dividing by number of batches
            log_dict[f"val_psnr_{inner_steps}_steps"] /= i + 1
            log_dict[f"val_loss_{inner_steps}_steps"] /= i + 1
            log_dict[f"val_loss_mse_{inner_steps}_steps"] /= i + 1
            log_dict[f"val_loss_stft_l1_{inner_steps}_steps"] /= i + 1
            log_dict[f"val_loss_stft_log_{inner_steps}_steps"] /= i + 1

            mean_psnr, mean_loss, mean_loss_mse, mean_loss_stft_l1, mean_loss_stft_log = (
                log_dict[f"val_psnr_{inner_steps}_steps"],
                log_dict[f"val_loss_{inner_steps}_steps"],
                log_dict[f"val_loss_mse_{inner_steps}_steps"],
                log_dict[f"val_loss_stft_l1_{inner_steps}_steps"],
                log_dict[f"val_loss_stft_log_{inner_steps}_steps"]
            )
            print(
                # f"Inner steps {inner_steps}, "
                f"Loss {mean_loss:.3f}, "
                f"Loss_MSE {mean_loss_mse:.3f}, "
                f"Loss_STFT_l1 {mean_loss_stft_l1:.3f}, "
                f"Loss_STFT_l1 {mean_loss_stft_log:.3f}, "
                f"PSNR {mean_psnr:.3f}"
            )

            # Use first setting of inner steps for best validation PSNR
            if inner_steps == self.args.validation_inner_steps[0]:
                if mean_psnr > self.best_val_psnr:
                    self.best_val_psnr = mean_psnr
                    # Optionally save new best model
                    if self.args.use_wandb and self.model_path:
                        torch.save(
                            {
                                "args": self.args,
                                "state_dict": self.func_rep.state_dict(),
                            },
                            self.model_path,
                        )

            if self.args.use_wandb:
                # Store final batch of reconstructions to visually inspect model
                # Shape (batch_size, channels, *spatial_dims)
                reconstruction = self.converter.to_data(
                    None, outputs["reconstructions"]
                )
                if self.patcher:
                    # If using patches, unpatch the reconstruction
                    # Shape (channels, *spatial_dims)
                    reconstruction = self.patcher.unpatch(reconstruction, spatial_shape)
                if self.converter.data_type == "mri":
                    # To store an image, slice MRI data along a single dimension
                    # Shape (1, depth, height, width) -> (1, height, width)
                    reconstruction = reconstruction[:, reconstruction.shape[1] // 2]

                if self.converter.data_type == "audio":
                    # Currently only support audio saving when using patches
                    if self.patcher:
                        # Unnormalize data from [0, 1] to [-1, 1] as expected by wandb
                        if self.test_dataloader.dataset.normalize:
                            reconstruction = 2 * reconstruction - 1
                        # Saved audio sample needs shape (num_samples, num_channels),
                        # so transpose
                        log_dict[
                            f"val_reconstruction_{inner_steps}_steps"
                        ] = wandb.Audio(
                            reconstruction.T.cpu(),
                            sample_rate=self.test_dataloader.dataset.sample_rate,
                        )

                if self.converter.data_type == 'time-series':
                    # so transpose
                    features_rec = self.converter.to_data(
                        None, features
                    )
                    if self.patcher:
                        features_rec = self.patcher.unpatch(features_rec, spatial_shape)
                    else:
                        features_rec = features_rec[0]
                        reconstruction = reconstruction[0]

                    # Unnormalize data from [0, 1] to [-1, 1] as expected by wandb
                    # if self.test_dataloader.dataset.normalize:
                    #     reconstruction = 2 * reconstruction - 1

                    # print(f'@ {outputs["reconstructions"].shape}')
                    # print(f'@ {reconstruction.shape}')
                    # print(f'@ {features.shape}')
                    # print(f'@ {features_rec.shape}')
                    # print(f'@ {features_rec.shape}')

                    if self.args.selected_features[0] == -1:
                        selected_features = list(range(features_rec.shape[0]))
                    else:
                        selected_features = self.args.selected_features
                    # print(f'@ {selected_features}')
                    # stft_avg_logging_l1 = {f'{fs}-{hs}-{wl}': 0 for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)}
                    # stft_avg_logging_log = {f'{fs}-{hs}-{wl}': 0 for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)}

                    for local_feat_idx, global_feat_idx in enumerate(selected_features[::-1]):
                        # print(f'feat_idx = {local_feat_idx}, actual_idx = {global_feat_idx}')

                        yhat = reconstruction[local_feat_idx].detach().cpu().flatten()
                        y = features_rec[local_feat_idx].detach().cpu().flatten()

                        # for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
                        #     stft_logging_l1, stft_logging_log  = losses.stft_metrics(y[None, ...], yhat[None, ...],
                        #                                                              fs, hs, wl,
                        #                                                              window=torch.hann_window(wl))

                            # stft_avg_logging_l1[f'{fs}-{hs}-{wl}'] += stft_logging_l1
                            # stft_avg_logging_log[f'{fs}-{hs}-{wl}'] += stft_logging_log

                            # log_dict[f'stft_l1_feat-{global_feat_idx}-fs{fs}-hs{hs}-wl{wl}'] = stft_logging_l1
                            # log_dict[f'stft_log_feat-{global_feat_idx}-fs{fs}-hs{hs}-wl{wl}'] = stft_logging_log

                        log_dict[f'loss_mse_feat-{global_feat_idx}'] = losses.batch_mse_fn(yhat[None,...], y[None,...]).mean()
                        log_dict[f'loss_stft_l1_feat-{global_feat_idx}'], \
                        log_dict[f'loss_stft_log_feat-{global_feat_idx}'] = self.stft_loss(reconstruction[local_feat_idx][None, ...],
                                                                                           features_rec[local_feat_idx][None, ...])

                        fig = plt.figure()

                        plt.plot(y, color='orange' ,alpha=0.8, label='ground truth feature')
                        plt.plot(yhat, color='blue' ,alpha=0.8, label='reconstruction')
                        plt.legend()
                        plt.ylabel('value')
                        plt.xlabel('time')
                        # table = wandb.Table(data=np.hstack([reconstruction.T.cpu().numpy().flatten(), features_rec.flatten()]),
                        #                     columns=['reconstruction', 'features_rec'])

                        log_dict[
                            f"val_reconstruction_steps_feat-{global_feat_idx}"
                        ] = plt
                        # plt.close(fig)

                    # for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
                    #     log_dict[f'stft_l1_avg-fs{fs}-hs{hs}-wl{wl}'] = stft_avg_logging_l1[f'{fs}-{hs}-{wl}']
                    #     log_dict[f'stft_log_avg-fs{fs}-hs{hs}-wl{wl}'] = stft_avg_logging_log[f'{fs}-{hs}-{wl}']

                else:
                    log_dict[f"val_reconstruction_steps"] = wandb.Image(
                        reconstruction
                    )

                wandb.log(log_dict, step=self.step)

        if not self.patcher:
            self.converter.coordinates = old_coordinates

        print("\n")
        return log_dict[f"val_psnr_{inner_steps}_steps"]
