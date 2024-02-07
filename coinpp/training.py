import coinpp.conversion as conversion
import coinpp.losses as losses
import coinpp.metalearning as metalearning
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
ALPHA = 0.7

def plot_ts(t, y, yhat, title, label1_start_idx, label1_end_idx):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    # fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=yhat, name='reconstruction', line=dict(color=f'rgba(70,130,180,{ALPHA})', width=2), mode='lines'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=y, name='ground truth', line=dict(color=f'rgba(255, 165, 0, {ALPHA})', width=2), mode='lines'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.sqrt((y - yhat)**2), name='error', line=dict(color=f'rgba(178,34,34, {ALPHA})', width=2), mode='lines+markers'), row=2, col=1)
    for s, e in zip(label1_start_idx, label1_end_idx):
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor="red", opacity=0.35,
            layer="below", line_width=0.05,
        ),
    fig.update_layout(title=title, xaxis_title='time', yaxis_title='value')
    return fig

class Trainer:
    def __init__(
        self,
        func_rep,
        converter,
        args,
        train_dataset,
        test_dataset,
        model_path="",
    ):
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

        self.outer_optimizer = torch.optim.Adam(
            self.func_rep.parameters(), lr=args.outer_lr
        )

        # self.full_anomaly_detection = args.full_anomaly_detection

        self.train_dataset = train_dataset
        self.train_monitor_dataset = test_dataset['train']
        self.val_dataset = test_dataset['val']
        self.test_dataset = test_dataset['test']

        self.train_monitor_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self._process_datasets()

        self.model_path = model_path
        self.step = 0
        self.epoch = 0
        self.best_val_psnr = 0.0

        if args.VanillaMAML:
            # TODO: Implement outer_step_MAML_chunked
            self.metalearner_outer_step = metalearning.outer_step_MAML
        else:
            self.metalearner_outer_step = metalearning.outer_step

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
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

        self.train_monitor_dataloader = torch.utils.data.DataLoader(
            self.train_monitor_dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

        if self.args.selected_features[0] == -1:
            self.selected_features = [i for i in range(self.train_dataset.data_selected_features.shape[0])]
        else:
            self.selected_features = self.args.selected_features

    def _checkpoint(self):
        """Save model checkpoint to wandb."""
        filepath_save = self.model_path.as_posix().format(self.step)
        torch.save({self.func_rep.state_dict()}, filepath_save)
        if self.args.wandb:
            wandb.save(filepath_save.name, base_path=wandb.run.dir, policy="live")

    def train_epoch(self, epoch):
        """Train model for a single epoch."""
        # start_time = time.time()
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

            outputs = self.metalearner_outer_step(
                self.func_rep,
                coordinates,
                features,
                inner_steps=self.args.inner_steps,
                inner_lr=self.args.inner_lr,
                is_train=True,
                return_reconstructions=False,
                gradient_checkpointing=self.args.gradient_checkpointing,
            )

            # Update parameters of base network
            self.outer_optimizer.zero_grad()
            outputs["loss"].backward(create_graph=False)
            self.outer_optimizer.step()

            log_dict = {"loss": outputs["loss"].item(), "psnr": outputs["psnr"]}

            self.step += 1

            if self.step % self.args.log_every == 0:
                print(
                    f'Step {self.step}, Loss {log_dict["loss"]:.3f}, PSNR {log_dict["psnr"]:.3f}'
                    # f'Step {self.step}, Loss {log_dict["loss"]:.3f}, PSNR {log_dict["psnr"]:.3f} time {time.time() - start_time:.3f}'
                )

            if self.args.use_wandb:
                wandb.log(log_dict, step=self.step)

        # if self.step % self.args.validate_every == 0 and self.step != 0:
        self.epoch = epoch
        if self.epoch % self.args.validate_every == 0:
            self.validation(dataloader=self.train_monitor_dataloader, label='train')
            self.validation(dataloader=self.val_dataloader, label='val')
            self.validation(dataloader=self.test_dataloader, label='test')


    def analyse(self, dataloader, label, inner_steps=1, plotting=True):

        # Initialize validation logging dict
        log_dict = {}

        if self.args.full_anomaly_detection:
            labels = dataloader.dataset.labels
            t = np.arange(len(labels))


        print(f'inner_steps {inner_steps}')
        log_dict[f"{label}_psnr_{inner_steps}_steps"] = 0.0
        log_dict[f"{label}_loss_{inner_steps}_steps"] = 0.0
        reconstructions_cat = None
        features_cat = None

        # Fit modulations for each validation datapoint
        for i, data in enumerate(dataloader):
            # print('$'*10)
            # print(i)
            # print(data.shape)
            data = data.to(self.args.device)


            old_coordinates = self.converter.coordinates
            self.converter.coordinates = None
            coordinates, features = self.converter.to_coordinates_and_features(data)

            outputs = self.metalearner_outer_step(
                self.func_rep,
                coordinates,
                features,
                inner_steps=inner_steps,
                inner_lr=self.args.validation_inner_lr if 'validation_inner_lr' in self.args else self.args.inner_lr,
                is_train=False,
                return_reconstructions=True,
                gradient_checkpointing=self.args.gradient_checkpointing,
            )

            log_dict[f"{label}_psnr_{inner_steps}_steps"] += outputs["psnr"]
            log_dict[f"{label}_loss_{inner_steps}_steps"] += outputs["loss"].item()

            if reconstructions_cat is not None:
                # print(reconstructions_cat.shape)
                # print(outputs['reconstructions'].shape)
                reconstructions_cat = torch.cat(
                    (reconstructions_cat, outputs['reconstructions']))
                features_cat = torch.cat((features_cat, features))
            else:
                reconstructions_cat = outputs['reconstructions']
                features_cat = features

        # Calculate average PSNR and loss by dividing by number of batches
        log_dict[f"{label}_psnr_{inner_steps}_steps"] /= i + 1
        log_dict[f"{label}_loss_{inner_steps}_steps"] /= i + 1

        mean_psnr, mean_loss = (
            log_dict[f"{label}_psnr_{inner_steps}_steps"],
            log_dict[f"{label}_loss_{inner_steps}_steps"],
        )
        print(
            f"Inner steps {inner_steps}, Loss {mean_loss:.3f}, PSNR {mean_psnr:.3f}"
        )


        # Store final batch of reconstructions to visually inspect model
        # Shape (batch_size, channels, *spatial_dims)
        reconstruction = self.converter.to_data(
            None, reconstructions_cat
        )

        if self.converter.data_type == "audio":
            # Currently only support audio saving when using patches
            if self.patcher:
                # Unnormalize data from [0, 1] to [-1, 1] as expected by wandb
                if self.test_dataloader.dataset.normalize:
                    reconstruction = 2 * reconstruction - 1
                # Saved audio sample needs shape (num_samples, num_channels),
                # so transpose
                log_dict[
                    f"{label}_reconstruction_{inner_steps}_steps"
                ] = wandb.Audio(
                    reconstruction.T.cpu(),
                    sample_rate=self.test_dataloader.dataset.sample_rate,
                )

        elif self.converter.data_type == 'time-series':
            # so transpose
            features_conv = self.converter.to_data(
                None, features_cat
            )

            # Concatenate batches
            features_conv = torch.cat([features_conv[i, :, :] for i in range(features_conv.shape[0])], dim=-1)
            reconstruction = torch.cat([reconstruction[i, :, :] for i in range(reconstruction.shape[0])],
                                       dim=-1)

            # Unnormalize data from [0, 1] to [-1, 1] as expected by wandb
            # if self.test_dataloader.dataset.normalize:
            #     reconstruction = 2 * reconstruction - 1

            # if self.args.selected_features[0] == -1:
            #     selected_features = list(range(features_rec.shape[0]))
            # else:
            #     selected_features = self.args.selected_features

            log_dict[f'y_feat'] = features_conv.detach().cpu().numpy().transpose(1, 0)
            log_dict[f'yhat_feat'] = reconstruction.detach().cpu().numpy().transpose(1, 0)

            for local_feat_idx in self.selected_features:

                yhat = reconstruction[local_feat_idx].detach().cpu().flatten()
                y = features_conv[local_feat_idx].detach().cpu().flatten()

                log_dict[f'{label}_loss_mse_{inner_steps}_steps_feat-{local_feat_idx}'] = losses.batch_mse_fn(yhat[None, ...],
                                                                                           y[None, ...]).mean()

                if plotting:
                    if not self.args.full_anomaly_detection:
                        fig, ax = plt.subplots()

                        ax.plot(y, color='orange', alpha=0.8, label='ground truth feature')
                        ax.plot(yhat, color='blue', alpha=0.8, label='reconstruction')
                        plt.legend()
                        plt.ylabel('value')
                        plt.xlabel('time')
                    else:
                        # print('Plot full anomaly detection')
                        fig = plot_ts(t=t, y=y, yhat=yhat,
                                      label1_start_idx=dataloader.dataset.label1_start_idx,
                                      label1_end_idx=dataloader.dataset.label1_end_idx,
                                      title=f"{label}_reconstruction_{inner_steps}_steps_feat-{local_feat_idx}")
                    # fig, ax = plt.subplots()
                    # ax.plot(t, y0, color='orange', alpha=0.8, label='ground truth feature')
                    # ax.plot(t, y1, color='red', alpha=0.8, label='ground truth feature anomalous')
                    # ax.plot(t, yhat, color='blue', alpha=0.8, label='reconstruction')
                    # plt.legend()
                    # plt.ylabel('value')
                    # plt.xlabel('time')

                # table = wandb.Table(data=np.hstack([reconstruction.T.cpu().numpy().flatten(), features_rec.flatten()]),
                #                     columns=['reconstruction', 'features_rec'])

                    log_dict[
                        f"{label}_reconstruction_{inner_steps}_steps_feat-{local_feat_idx}"
                    ] = fig
                # plt.close(fig)

        else:
            log_dict[f"{label}_reconstruction_{inner_steps}_steps"] = wandb.Image(
                reconstruction
            )
        self.converter.coordinates = old_coordinates

        return log_dict


    def validation(self, dataloader, label='val'):
        """Run trained model on validation dataset."""
        print(f"\nValidation, Epoch {self.epoch:>6d} ----- Step {self.step:>6d} ----- label {label}:")
        old_coordinates = None

        # If num_validation_points is -1, validate on entire validation dataset,
        # otherwise validate on a subsample of points
        full_validation = self.args.num_validation_points == -1
        num_validation_batches = self.args.num_validation_points // self.args.batch_size

        # Initialize validation logging dict
        log_dict = {}

        if self.args.full_anomaly_detection:
            labels = dataloader.dataset.labels
            t = np.arange(len(labels))
            arr_cond_1 = labels == 1
            arr_cond_0 = labels == 0

        # Save model
        # self._checkpoint()

        # Evaluate model for different numbers of inner loop steps
        for inner_steps in self.args.validation_inner_steps:
            print(f'inner_steps {inner_steps}')
            log_dict[f"{label}_psnr_{inner_steps}_steps"] = 0.0
            log_dict[f"{label}_loss_{inner_steps}_steps"] = 0.0
            reconstructions_cat = None
            features_cat = None

            # Fit modulations for each validation datapoint
            for i, data in enumerate(dataloader):

                data = data.to(self.args.device)

                old_coordinates = self.converter.coordinates
                self.converter.coordinates = None
                coordinates, features = self.converter.to_coordinates_and_features(
                    data
                )

                outputs = self.metalearner_outer_step(
                    self.func_rep,
                    coordinates,
                    features,
                    inner_steps=inner_steps,
                    inner_lr=self.args.validation_inner_lr if 'validation_inner_lr' in self.args else self.args.inner_lr,
                    is_train=False,
                    return_reconstructions=True,
                    gradient_checkpointing=self.args.gradient_checkpointing,
                )

                log_dict[f"{label}_psnr_{inner_steps}_steps"] += outputs["psnr"]
                log_dict[f"{label}_loss_{inner_steps}_steps"] += outputs["loss"].item()

                if reconstructions_cat is not None:
                    # print(reconstructions_cat.shape)
                    # print(outputs['reconstructions'].shape)
                    reconstructions_cat = torch.cat(
                        (reconstructions_cat, outputs['reconstructions']))
                    features_cat = torch.cat((features_cat, features))
                else:
                    reconstructions_cat = outputs['reconstructions']
                    features_cat = features

                # print(f"outputs_cat.shape = {reconstructions_cat.shape}")
                # print(f"features_cat.shape = {features_cat.shape}")

                if not full_validation and i >= num_validation_batches - 1:
                    break

            # Calculate average PSNR and loss by dividing by number of batches
            log_dict[f"{label}_psnr_{inner_steps}_steps"] /= i + 1
            log_dict[f"{label}_loss_{inner_steps}_steps"] /= i + 1

            mean_psnr, mean_loss = (
                log_dict[f"{label}_psnr_{inner_steps}_steps"],
                log_dict[f"{label}_loss_{inner_steps}_steps"],
            )
            print(
                f"Inner steps {inner_steps}, Loss {mean_loss:.3f}, PSNR {mean_psnr:.3f}"
            )

            # Use first setting of inner steps for best validation PSNR
            if self.args.save_models == "best_val" and label == "val":
                if inner_steps == self.args.validation_inner_steps[0]:
                    if mean_psnr > self.best_val_psnr:
                        self.best_val_psnr = mean_psnr
                        # Optionally save new best model
                        if self.args.use_wandb and self.model_path:
                            filepath_save = self.model_path.as_posix().format(self.epoch)
                            print(f'Saving model to wandb at {filepath_save} because best val psnr is {self.best_val_psnr}')
                            torch.save(
                                {
                                    "args": self.args,
                                    "state_dict": self.func_rep.state_dict(),
                                },
                                filepath_save,
                            )

            elif self.args.save_models == 'all' and label == 'val':
                if self.args.use_wandb:
                    filepath_save = self.model_path.as_posix().format(self.epoch)
                    print(f'Saving model to wandb at {filepath_save}')
                    torch.save(
                        {
                            "args": self.args,
                            "state_dict": self.func_rep.state_dict(),
                        },
                        filepath_save,
                    )
            else:
                print('Not saving model')

            if self.args.use_wandb:
                # Store final batch of reconstructions to visually inspect model
                # Shape (batch_size, channels, *spatial_dims)
                reconstruction = self.converter.to_data(
                    None, reconstructions_cat
                )
                # print(f'2 recostruction.shape = {reconstruction.shape}')

                if self.converter.data_type == "audio":
                    # Currently only support audio saving when using patches
                    if self.patcher:
                        # Unnormalize data from [0, 1] to [-1, 1] as expected by wandb
                        if self.test_dataloader.dataset.normalize:
                            reconstruction = 2 * reconstruction - 1
                        # Saved audio sample needs shape (num_samples, num_channels),
                        # so transpose
                        log_dict[
                            f"{label}_reconstruction_{inner_steps}_steps"
                        ] = wandb.Audio(
                            reconstruction.T.cpu(),
                            sample_rate=self.test_dataloader.dataset.sample_rate,
                        )

                elif self.converter.data_type == 'time-series':
                    # so transpose
                    features_conv = self.converter.to_data(
                        None, features_cat
                    )
                    # print(f'1 features_rec.shape = {features_rec.shape}')
                    # print(f'1 reconstruction.shape = {reconstruction.shape}')

                    # Concatenate batches
                    features_conv = torch.cat([features_conv[i, :, :] for i in range(features_conv.shape[0])], dim=-1)
                    reconstruction = torch.cat([reconstruction[i, :, :] for i in range(reconstruction.shape[0])], dim=-1)
                    # print(f'3 no patcher features_rec.shape = {features_rec.shape}')
                    # print(f'3 no patcher recostruction.shape = {reconstruction.shape}')

                    # Unnormalize data from [0, 1] to [-1, 1] as expected by wandb
                    # if self.test_dataloader.dataset.normalize:
                    #     reconstruction = 2 * reconstruction - 1

                    # print(f'@ {outputs["reconstructions"].shape}')
                    # print(f'@ {reconstruction.shape}')
                    # print(f'@ {features.shape}')
                    # print(f'@ {features_rec.shape}')

                    # if self.args.selected_features[0] == -1:
                    #     selected_features = list(range(features_rec.shape[0]))
                    # else:
                    #     selected_features = self.args.selected_features
                    # print(f'@ {selected_features}')
                    for local_feat_idx, global_feat_idx in enumerate(self.selected_features[::-1]):
                        # print(f'feat_idx = {local_feat_idx}, actual_idx = {global_feat_idx}')

                        yhat = reconstruction[local_feat_idx].detach().cpu().flatten()
                        y = features_conv[local_feat_idx].detach().cpu().flatten()

                        # for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
                        #     stft_logging_l1, stft_logging_log  = losses.stft_metrics(y[None, ...], yhat[None, ...],
                        #                                                              fs, hs, wl,
                        #                                                              window=torch.hann_window(wl))

                            # stft_avg_logging_l1[f'{fs}-{hs}-{wl}'] += stft_logging_l1
                            # stft_avg_logging_log[f'{fs}-{hs}-{wl}'] += stft_logging_log

                            # log_dict[f'stft_l1_feat-{global_feat_idx}-fs{fs}-hs{hs}-wl{wl}'] = stft_logging_l1
                            # log_dict[f'stft_log_feat-{global_feat_idx}-fs{fs}-hs{hs}-wl{wl}'] = stft_logging_log

                        log_dict[f'{label}_loss_mse_feat-{global_feat_idx}'] = losses.batch_mse_fn(yhat[None,...], y[None,...]).mean()
                        # log_dict[f'loss_stft_l1_feat-{global_feat_idx}'], \
                        # log_dict[f'loss_stft_log_feat-{global_feat_idx}'] = self.stft_loss(reconstruction[local_feat_idx][None, ...],
                        #                                                                    features_rec[local_feat_idx][None, ...])
                        if self.args.plot_wandb:
                            if not self.args.full_anomaly_detection:
                                fig, ax  = plt.subplots()

                                ax.plot(y, color='orange' ,alpha=0.8, label='ground truth feature')
                                ax.plot(yhat, color='blue' ,alpha=0.8, label='reconstruction')
                                plt.legend()
                                plt.ylabel('value')
                                plt.xlabel('time')

                            else:
                                # print('Plot full anomaly detection')
                                fig = plot_ts(t=t, y=y, yhat=yhat,
                                              label1_start_idx=dataloader.dataset.label1_start_idx,
                                              label1_end_idx=dataloader.dataset.label1_end_idx,
                                              title=f"{label}_reconstruction_{inner_steps}_steps_feat-{global_feat_idx}")
                            # print(f'Plotting {label}_reconstruction_{inner_steps}_steps_feat-{global_feat_idx}')
                            log_dict[f"{label}_reconstruction_{inner_steps}_steps_feat-{global_feat_idx}"] = fig

                else:
                    raise ValueError('Converter.data_type not supported. Allowed values are "audio" and "time-series"')
                wandb.log(log_dict, step=self.step)

        self.converter.coordinates = old_coordinates

        print("\n")
