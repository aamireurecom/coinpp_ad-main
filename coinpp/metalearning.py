import coinpp.losses as losses
import torch
import torch.utils.checkpoint as cp
import copy
import functorch
import time


def inner_loop(
    func_rep,
    modulations,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.

    Args:
        func_rep (models.ModulatedSiren):
        modulations (torch.Tensor): Shape (batch_size, latent_dim).
        coordinates (torch.Tensor): Coordinates at which function representation
            should be evaluated. Shape (batch_size, *, coordinate_dim).
        features (torch.Tensor): Target features for model to match. Shape
            (batch_size, *, feature_dim).
        inner_steps (int): Number of inner loop steps to take.
        inner_lr (float): Learning rate for inner loop.
        is_train (bool):
        gradient_checkpointing (bool): If True uses gradient checkpointing. This
            can massively reduce memory consumption.
    """
    fitted_modulations = modulations
    for step in range(inner_steps):
        if gradient_checkpointing:
            fitted_modulations = cp.checkpoint(
                inner_loop_step,
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                torch.as_tensor(inner_lr),
                torch.as_tensor(is_train),
                torch.as_tensor(gradient_checkpointing),
            )
        else:
            fitted_modulations = inner_loop_step(
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                inner_lr,
                is_train,
                gradient_checkpointing,
            )
    return fitted_modulations


def inner_loop_step(
    func_rep,
    modulations,
    coordinates,
    features,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
):
    """Performs a single inner loop step."""
    detach = not torch.is_grad_enabled() and gradient_checkpointing
    batch_size = len(features)

    with torch.enable_grad():
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch
        loss = losses.mse_fn(features_recon, features) * batch_size
        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]

    # Perform single gradient descent step
    return modulations - inner_lr * grad


def outer_step(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """
    func_rep.zero_grad()
    batch_size = len(coordinates)

    modulations_init = torch.zeros(
        batch_size, func_rep.modulation_net.latent_dim, device=coordinates.device
    ).requires_grad_()

    # Run inner loop
    modulations = inner_loop(
            func_rep,
            modulations_init,
            coordinates,
            features,
            inner_steps,
            inner_lr,
            is_train,
            gradient_checkpointing,
        )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        # While the loss in the inner loop is individual for each set of
        # modulations, the loss in the outer loop does depend on the entire
        # batch (we update the base network such that all modulations can easily
        # be fit). We therefore take the mean across the batch dimension so the
        # loss is invariant to the number of elements in the batch
        # Shape (batch_size,)
        per_example_loss = losses.batch_mse_fn(features_recon, features)
        # Shape (1,)
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = features_recon

    return outputs


def outer_step_MAML(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
):

    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """
    # t0 = time.time()

    func_rep.zero_grad()

    fmodel, _, buffers = functorch.make_functional_with_buffers(func_rep, disable_autograd_tracking=False)

    base_params = list(func_rep.parameters())


    # Run inner loop
    reconstruction, loss_outer = inner_loop_MAML(
        fmodel,
        base_params,
        coordinates,
        features,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        )

    outputs = {
        "loss": loss_outer,
        "psnr": losses.mse2psnr(loss_outer).item(),
        "modulations": None,
    }

    if return_reconstructions:
        outputs["reconstructions"] = reconstruction

    # t1 = time.time() - t0
    # print(f'time {t1:0.5f} s')

    return outputs

def inner_loop_MAML(
        fmodel,
        base_params,
        coordinates,
        features,
        inner_steps,
        inner_lr,
        is_train=False,
        gradient_checkpointing=False,
):

    loss_fn = functorch.vmap(losses.loss_functional, in_dims=(0, None, None, 0, 0))
    fmodel_vmap = functorch.vmap(fmodel, in_dims=(0, None, 0))
    grad_fn = functorch.grad(losses.loss_functional)
    params = [p.clone()[None, ...].repeat(coordinates.shape[0], *(1 for _ in p.shape)) for p in base_params]

    for step in range(inner_steps):
        loss_ = loss_fn(params, fmodel, tuple(), coordinates, features)

        for i, (x, y) in enumerate(zip(coordinates, features)):
            params_to_update = [p[i] for p in params]
            grads = grad_fn(params_to_update, fmodel, tuple(), x, y)

            for p, g in zip(params_to_update, grads):
                p.data -= inner_lr * g

    reconstructions = fmodel_vmap(params, tuple(), coordinates)

    # print(f'reconstructions_inner[0].shape {reconstructions[0].shape}')
    return reconstructions, loss_.mean()


def inner_loop_step_MAML(
    func_inner,
    func_grad,
    compute_loss,
    params_inner,
    coordinates,
    features,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
):
    """Performs a single inner loop step."""

    # detach = not torch.is_grad_enabled() and gradient_checkpointing
    # batch_size = len(features)

    with torch.enable_grad():
        features_recon = func_inner(params_inner, coordinates)
        loss = compute_loss(params_inner, coordinates, features)
        grads = func_grad(params_inner, coordinates, features)

    params_inner = [p - inner_lr * g for p, g in zip(params_inner, grads)]
    return params_inner, features_recon, loss
    #     # Note we multiply by batch size here to undo the averaging across batch
    #     # elements from the MSE function. Indeed, each set of modulations is fit
    #     # independently and the size of the gradient should not depend on how
    #     # many elements are in the batch
    #     loss = losses.mse_fn(features_recon, features) * batch_size
    #     # If we are training, we should create graph since we will need this to
    #     # compute second order gradients in the MAML outer loop
    #     grad = torch.autograd.grad(
    #         loss,
    #         func_inner.parameters(),
    #         create_graph=is_train and not detach,
    #         # retain_graph=True,
    #         # allow_unused=True
    #     )
    #
    # # print(f'grad[0]: {grad[0]}')
    # # print(grad)
    #
    # # Perform single gradient descent step
    # for i, param in enumerate(func_rep.parameters()):
    #     # print('i' * 10 + f'{i}')
    #     # print(param)
    #     param.data = param.data - inner_lr * grad[i]
    #
    # return loss