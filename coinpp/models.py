# Based on https://github.com/EmilienDupont/coin
import torch
from torch import nn
import math

# from torch import sqrt
from torch.nn import Linear
from torch import Tensor
from torch.nn import Parameter, init
from torch.nn import functional as F



class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Wire(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0=1.0, s0=0):
        super().__init__()
        self.w0 = w0
        self.s0 = s0

    def forward(self, x):
        return torch.sin(self.w0 * x) * torch.exp(-(self.s0 * x) ** 2)

class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        is_last (bool): Whether this is last layer of model. If it is, no
            activation is applied and 0.5 is added to the output. Since we
            assume all training data lies in [0, 1], this allows for centering
            the output of the model.
        use_bias (bool): Whether to learn bias in linear layer.
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        s0=0.0,
        c=6.0,
        is_first=False,
        is_last=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (math.sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Wire(w0, s0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        if self.is_last:
            # We assume target data is in [0, 1], so adding 0.5 allows us to learn
            # zero-centered features
            out += 0.5
        else:
            out = self.activation(out)
        return out


class Siren(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        s0=0.0,
        w0_initial=30.0,
        use_bias=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers

        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(
                SirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    s0=s0,
                    use_bias=use_bias,
                    is_first=is_first,
                    activation=None,
                )
            )

        self.net = nn.Sequential(*layers)

        self.last_layer = SirenLayer(
            dim_in=dim_hidden, dim_out=dim_out, w0=w0, s0=s0, use_bias=use_bias, is_last=True
        )
        # self.last_last_layer = SirenLayer(
        #     dim_in=dim_out, dim_out=dim_out, w0=w0, use_bias=use_bias, is_last=True, activation=lambda x: x
        # )

    def forward(self, x):
        """Forward pass of SIREN model.

        Args:
            x (torch.Tensor): Tensor of shape (*, dim_in), where * means any
                number of dimensions.

        Returns:
            Tensor of shape (*, dim_out).
        """
        x = self.net(x)
        x = self.last_layer(x)
        # x = self.last_last_layer(x)
        return x


class SirenResidualBlock(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_hidden,
            dim_out,
            w0=30.0,
            s0=0.0,
            w0_initial=30.0,
            use_bias=True,
            is_first=False,
    ):
        super().__init__()

        self.l1 = SirenLayer(
            dim_in=dim_in,
            dim_out=dim_hidden,
            w0=w0,
            s0=s0,
            use_bias=use_bias,
            is_first=is_first,
            activation=None,
        )
        self.l2 = SirenLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            s0=s0,
            use_bias=use_bias,
            is_first=False,
            activation=None,
        )

    def forward(self, x):
        h = self.l1(x)
        h = self.l2(h)
        return x + h

class SirenResidual(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        s0=0.0,
        w0_initial=30.0,
        use_bias=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers

        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(
                SirenLayer(
                    is_first=False,
                    is_last=False,
                    use_bias=True,
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    s0=s0,
                    activation=None,
                )
            )

        self.net = nn.Sequential(*layers)

        self.last_layer = SirenLayer(
            dim_in=dim_hidden, dim_out=dim_out, w0=w0, s0=s0, use_bias=use_bias, is_last=True
        )
        # self.last_last_layer = SirenLayer(
        #     dim_in=dim_out, dim_out=dim_out, w0=w0, use_bias=use_bias, is_last=True, activation=lambda x: x
        # )

    def forward(self, x):
        """Forward pass of SIREN model.

        Args:
            x (torch.Tensor): Tensor of shape (*, dim_in), where * means any
                number of dimensions.

        Returns:
            Tensor of shape (*, dim_out).
        """
        h = None
        for i, layer in enumerate(self.net, start=1):
            print(i)
            if i % 2:
                print('odd')
                h = layer(x)
                print(h.shape, x.shape)
            else:
                print('even')
                x = layer(h) + x
                print(h.shape, x.shape)
        # x = self.net(x)
        # x = self.last_layer(x)
        # x = self.last_last_layer(x)
        return self.last_layer(x)



class SirenParallel(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        s0=0.0,
        w0_initial=30.0,
        use_bias=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.net_parallel = nn.ModuleList()
        self.last_layer_parallel = nn.ModuleList()

        for _ in range(dim_out):

            layers = []
            for ind in range(num_layers - 1):
                is_first = ind == 0
                layer_w0 = w0_initial if is_first else w0
                layer_dim_in = dim_in if is_first else dim_hidden

                layers.append(
                    SirenLayer(
                        is_first=False,
                        is_last=False,
                        use_bias=True,
                        dim_in=layer_dim_in,
                        dim_out=dim_hidden,
                        w0=layer_w0,
                        s0=s0,
                        activation=None,
                    )
                )

            self.net_parallel.append(nn.Sequential(*layers))
            self.last_layer_parallel.append(SirenLayer(
                dim_in=dim_hidden, dim_out=1, w0=w0, s0=s0, use_bias=use_bias, is_last=True
            ))

        # self.last_last_layer = SirenLayer(
        #     dim_in=dim_out, dim_out=dim_out, w0=w0, use_bias=use_bias, is_last=True, activation=lambda x: x
        # )

    def forward(self, x):
        """Forward pass of SIREN model.

        Args:
            x (torch.Tensor): Tensor of shape (*, dim_in), where * means any
                number of dimensions.

        Returns:
            Tensor of shape (*, dim_out).
        """
        list_out = []
        for subnet, last_layer in zip(self.net_parallel, self.last_layer_parallel):
            # print(x.shape)
            h = subnet(x)
            # print(h.shape)
            list_out.append(last_layer(h))
        # print(list_out)
        return torch.cat(list_out, dim=1)

class LinearParallel(Linear):
    def __init__(self, in_block: int, out_block: int, num_blocks:int, bias: bool = True,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.A = torch.block_diag(*[torch.ones((out_block, in_block)) for _ in range(num_blocks)]).to(device)
        self.in_block = in_block
        self.out_block = out_block
        self.num_blocks = num_blocks
        self.in_features = in_block * num_blocks
        self.out_features = out_block * num_blocks
        self.weight = Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        fake_weight = torch.empty(self.out_block, self.in_block)
        fan_in, _ = init._calculate_fan_in_and_fan_out(fake_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.weight, -bound, bound)
        # init.kaiming_uniform_(, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(fake_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # print(f'out.shape = {out.shape}')
        return F.linear(input, self.weight * self.A, self.bias)

    def extra_repr(self) -> str:
        return 'in_block={}, out_block={}, num_blocks={}, bias={}, A.shape={}'.format(
            self.in_block, self.out_block, self.num_blocks, self.bias is not None, self.A.shape
        )

class LinearParallelBMM(Linear):
    def __init__(self, in_block: int, out_block: int, num_blocks: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        # self.A = torch.block_diag(*[torch.ones((out_block, in_block)) for i in range(num_blocks)])
        self.in_block = in_block
        self.out_block = out_block
        self.num_blocks = num_blocks
        self.in_features = in_block * num_blocks
        self.out_features = out_block * num_blocks
        self.weight = Parameter(torch.empty((num_blocks, self.out_block, self.in_block), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty((num_blocks, out_block), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        fake_weight = torch.empty(self.out_block, self.in_block)
        fan_in, _ = init._calculate_fan_in_and_fan_out(fake_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.weight, -bound, bound)
        # init.kaiming_uniform_(, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(fake_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # B = input.shape[0]*input.shape[1]
        # N = self.num_blocks
        # M = self.out_block
        # V = self.in_block
        # print(f'self.weight.shape, input.shape, self.bias.shape, self.num_blocks, self.out_block, self.in_block'
        #       f'= {self.weight.shape}, {input.shape}, {self.bias.shape}, {self.num_blocks}, {self.out_block}, {self.in_block}')
        # print(f'A.shape, B.shape = {A.shape}, {B.shape}')
        return (torch.matmul(self.weight.expand(input.shape[0]*input.shape[1], -1, -1, -1).reshape(input.shape[0]*input.shape[1] * self.num_blocks, self.out_block, self.in_block),
                            input.reshape(input.shape[0]*input.shape[1] * self.num_blocks, self.in_block, 1)).reshape(input.shape[0]*input.shape[1], self.num_blocks, self.out_block) + self.bias.expand(input.shape[0]*input.shape[1], -1, -1)).squeeze().reshape(input.shape[0], input.shape[1], -1)
        # return torch.bmm(input, self.weight) + self.bias
        # return F.linear(input, self.weight * self.A, self.bias)

    def extra_repr(self) -> str:
        return 'in_block={}, out_block={}, num_blocks={}, bias={}'.format(
            self.in_block, self.out_block, self.num_blocks, self.bias is not None
        )

class SirenLayerParallelMatrix(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        is_last (bool): Whether this is last layer of model. If it is, no
            activation is applied and 0.5 is added to the output. Since we
            assume all training data lies in [0, 1], this allows for centering
            the output of the model.
        use_bias (bool): Whether to learn bias in linear layer.
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        num_blocks,
        w0=30.0,
        s0=0.0,
        c=6.0,
        is_first=False,
        is_last=False,
        use_bias=True,
        activation=None,
        BMM=False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.is_last = is_last
        if not BMM:
            self.linear = LinearParallel(in_block=dim_in, out_block=dim_out, num_blocks=num_blocks, bias=use_bias)
        else:
            self.linear = LinearParallelBMM(in_block=dim_in, out_block=dim_out, num_blocks=num_blocks, bias=use_bias)
        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (math.sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Wire(w0, s0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        if self.is_last:
            # We assume target data is in [0, 1], so adding 0.5 allows us to learn
            # zero-centered features
            out += 0.5
        else:
            out = self.activation(out)
        return out

class SirenParallelMatrix(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        s0=0.0,
        w0_initial=30.0,
        use_bias=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers

        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(
                SirenLayerParallelMatrix(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    num_blocks=dim_out,
                    w0=layer_w0,
                    s0=s0,
                    use_bias=use_bias,
                    is_first=is_first,
                    activation=None,
                )
            )

        self.net = nn.Sequential(*layers)

        self.last_layer = SirenLayerParallelMatrix(
            dim_in=dim_hidden, dim_out=1, num_blocks=dim_out, w0=w0, s0=s0, use_bias=use_bias, is_last=True
        )
        # self.last_last_layer = SirenLayer(
        #     dim_in=dim_out, dim_out=dim_out, w0=w0, use_bias=use_bias, is_last=True, activation=lambda x: x
        # )

    def forward(self, x):
        """Forward pass of SIREN model.

        Args:
            x (torch.Tensor): Tensor of shape (*, dim_in), where * means any
                number of dimensions.

        Returns:
            Tensor of shape (*, dim_out).
        """
        x = torch.cat([x] * self.dim_out, dim=-1)
        x = self.net(x)
        x = self.last_layer(x)
        # x = self.last_last_layer(x)
        return x


class LayerIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = self.activation = nn.Identity()

    def forward(self, x):
        return self.linear(x)

class SirenMultiBranch(nn.Module):
    # TODO: Review doc
    """SIREN multibranch model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden_trunk,
        dim_hidden_branch,
        dim_out,
        num_layers_trunk,
        num_layers_branch,
        w0=30.0,
        s0=0.0,
        w0_initial=30.0,
        use_bias=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden_branch = dim_hidden_branch
        self.dim_hidden_trunk = dim_hidden_trunk
        self.dim_out = dim_out
        self.num_layers_branch = num_layers_branch
        self.num_layers_trunk = num_layers_trunk
        self.w0 = w0
        self.w0_initial = w0_initial
        self.use_bias = use_bias

        if num_layers_trunk > 0:
            layers = self._subnet(num_layers=num_layers_trunk, first_dim_in=dim_in, w0_initial=w0_initial, w0=w0, s0=s0,
                              dim_hidden=dim_hidden_trunk, use_bias=use_bias)


        else:
            layers = [LayerIdentity()]

        self.net_trunk = nn.Sequential(*layers)

        self.list_net_branch = nn.ModuleList()

        for i in range(dim_out):
            layers = self._subnet(num_layers=num_layers_branch - 1, first_dim_in=dim_hidden_trunk, w0_initial=w0, w0=w0, s0=s0,
                                  dim_hidden=dim_hidden_branch, use_bias=use_bias)

            last_layer = SirenLayer(dim_in=dim_hidden_branch,
                                    dim_out=1,
                                    w0=w0,
                                    s0=s0,
                                    use_bias=use_bias,
                                    is_last=True)
            layers += [last_layer]
            net_branch = nn.Sequential(*layers)

            self.list_net_branch.append(net_branch)

    def _subnet(self, num_layers, first_dim_in, dim_hidden, w0_initial, w0, s0, use_bias):
        layers = []

        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = first_dim_in if is_first else dim_hidden

            layers.append(
                SirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    s0=s0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        return layers
    def forward(self, x):
        # TODO: Review doc
        """Forward pass of SIREN model.

        Args:
            x (torch.Tensor): Tensor of shape (*, dim_in), where * means any
                number of dimensions.

        Returns:
            Tensor of shape (*, dim_out).
        """
        out_branch = [] # torch.zeros(self.dim_out)
        out_trunk = self.net_trunk(x)

        for branch in self.list_net_branch:
            # out_branch[idx_branch] = self.list_net_branch[idx_branch](out_trunk)
            out_branch.append(branch(out_trunk))

        output = torch.cat(out_branch, -1)
        return output


class SirenMultiBranchListw0(nn.Module):
    # TODO: Review doc
    """SIREN multibranch model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden_trunk,
        dim_hidden_branch,
        dim_out,
        num_layers_trunk,
        num_layers_branch,
        list_w0_branch=(30.0,),
        w0_trunk=30.0,
        w0_initial=30.0,
        use_bias=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden_branch = dim_hidden_branch
        self.dim_hidden_trunk = dim_hidden_trunk
        self.dim_out = dim_out
        self.num_layers_branch = num_layers_branch
        self.num_layers_trunk = num_layers_trunk
        self.w0 = w0_trunk
        self.w0_initial = w0_initial
        self.use_bias = use_bias

        layers = self._subnet(num_layers=num_layers_trunk, first_dim_in=dim_in, w0_initial=w0_initial, w0=w0_trunk,
                              dim_hidden=dim_hidden_trunk, use_bias=use_bias)

        self.net_trunk = nn.Sequential(*layers)

        self.list_net_branch = nn.ModuleList()

        for i in range(dim_out):
            layers = self._subnet(num_layers=num_layers_branch - 1, first_dim_in=dim_hidden_trunk, w0_initial=w0_trunk, w0=list_w0_branch[i],
                                  dim_hidden=dim_hidden_branch, use_bias=use_bias)

            last_layer = SirenLayer(dim_in=dim_hidden_branch,
                                    dim_out=1,
                                    w0=list_w0_branch[i],
                                    use_bias=use_bias,
                                    is_last=True)
            layers += [last_layer]
            net_branch = nn.Sequential(*layers)

            self.list_net_branch.append(net_branch)

    def _subnet(self, num_layers, first_dim_in, dim_hidden, w0_initial, w0, use_bias):
        layers = []

        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = first_dim_in if is_first else dim_hidden

            layers.append(
                SirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        return layers
    def forward(self, x):
        # TODO: Review doc
        """Forward pass of SIREN model.

        Args:
            x (torch.Tensor): Tensor of shape (*, dim_in), where * means any
                number of dimensions.

        Returns:
            Tensor of shape (*, dim_out).
        """
        out_branch = [] # torch.zeros(self.dim_out)
        out_trunk = self.net_trunk(x)

        for branch in self.list_net_branch:
            # out_branch[idx_branch] = self.list_net_branch[idx_branch](out_trunk)
            out_branch.append(branch(out_trunk))

        output = torch.cat(out_branch, -1)
        return output


class ModulatedSiren(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        s0=0.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        # modulate_w0=False,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        last_layer_modulated=False,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            s0,
            w0_initial,
            use_bias,
        )

        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        # self.modulate_w0 = modulate_w0
        self.w0 = w0
        self.s0 = s0
        self.w0_initial = w0_initial
        self.last_layer_modulated = last_layer_modulated

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if last_layer_modulated:
            num_modulations += dim_out

        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        # if self.modulate_w0:
        #     num_modulations += num_layers - 1

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        # print(self.__dict__)

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx : idx + self.dim_hidden].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[
                    :, mid_idx + idx : mid_idx + idx + self.dim_hidden
                ].unsqueeze(1)
            else:
                shift = 0.0

            x = module.linear(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.dim_hidden

        if self.last_layer_modulated:
            if self.modulate_scale:
                scale = modulations[:, idx : idx + self.dim_out].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                shift = modulations[
                    :, mid_idx + idx : mid_idx + idx + self.dim_out].unsqueeze(1)
            else:
                shift = 0.0

            x = self.last_layer.linear(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            out = x + 0.5
        else:
            # Shape (batch_size, num_points, dim_out)
            out = self.last_layer(x)

        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenParallel(SirenParallel):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        s0=0.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        # modulate_w0=False,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        last_layer_modulated=False,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            s0,
            w0_initial,
            use_bias,
        )

        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        # self.modulate_w0 = modulate_w0
        self.w0 = w0
        self.s0 = s0
        self.w0_initial = w0_initial
        self.last_layer_modulated = last_layer_modulated

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_out * dim_hidden * (num_layers - 1)
        if last_layer_modulated:
            num_modulations += dim_out

        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        # if self.modulate_w0:
        #     num_modulations += num_layers - 1

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        print(self.__dict__)

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        x_start = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )

        idx = 0
        list_out = []
        for subnet, last_layer in zip(self.net_parallel, self.last_layer_parallel):
            for m, module in enumerate(subnet):
                # print(f"subnet {c} layer {m}")
                if self.modulate_scale:
                    # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                    # modulations remain zero centered
                    scale = modulations[:, idx : idx + self.dim_hidden].unsqueeze(1) + 1.0
                else:
                    scale = 1.0

                if self.modulate_shift:
                    # Shape (batch_size, 1, dim_hidden)
                    shift = modulations[
                        :, mid_idx + idx : mid_idx + idx + self.dim_hidden
                    ].unsqueeze(1)
                else:
                    shift = 0.0
                if m == 0:
                    x = module.linear(x_start)
                else:
                    x = module.linear(x)
                # x = module.linear(x_start)
                x = scale * x + shift  # Broadcast scale and shift across num_points
                x = module.activation(x)  # (batch_size, num_points, dim_hidden)

                idx = idx + self.dim_hidden

            if self.last_layer_modulated:
                if self.modulate_scale:
                    scale = modulations[:, idx : idx + self.dim_out].unsqueeze(1) + 1.0
                else:
                    scale = 1.0

                if self.modulate_shift:
                    shift = modulations[
                        :, mid_idx + idx : mid_idx + idx + self.dim_out].unsqueeze(1)
                else:
                    shift = 0.0

                x = last_layer.linear(x)
                x = scale * x + shift  # Broadcast scale and shift across num_points
                out = x + 0.5
            else:
                # Shape (batch_size, num_points, dim_out)
                out = last_layer(x)
            # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
            list_out.append(out.view(*x_shape, out.shape[-1]))
        # Concatenate all parallel branches
        return torch.cat(list_out, dim=-1)


class ModulatedSirenParallelMatrix(SirenParallelMatrix):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        s0=0.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        # modulate_w0=False,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        last_layer_modulated=False,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            s0,
            w0_initial,
            use_bias,
        )

        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        # self.modulate_w0 = modulate_w0
        self.w0 = w0
        self.s0 = s0
        self.w0_initial = w0_initial
        self.last_layer_modulated = last_layer_modulated

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_out * dim_hidden * (num_layers - 1)
        if last_layer_modulated:
            num_modulations += dim_out

        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        # if self.modulate_w0:
        #     num_modulations += num_layers - 1

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        print(self.__dict__)

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # print(f'1) x.shape {x.shape}')
        x = torch.cat([x] * self.dim_out, dim=-1)
        # print(f'2) x.shape {x.shape}')

        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        x = x.view(x.shape[0], -1, x.shape[-1])
        # print(f'3) x.shape {x.shape}')

        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )

        idx = 0
        for i, module in enumerate(self.net):
            # print(f"subnet {c} layer {m}")
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx : idx + self.dim_hidden * self.dim_out].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[
                    :, mid_idx + idx : mid_idx + idx + self.dim_hidden * self.dim_out
                ].unsqueeze(1)
            else:
                shift = 0.0

            x = module.linear(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.dim_hidden * self.dim_out

        if self.last_layer_modulated:
            if self.modulate_scale:
                scale = modulations[:, idx : idx + self.dim_out].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                shift = modulations[
                    :, mid_idx + idx : mid_idx + idx + self.dim_out].unsqueeze(1)
            else:
                shift = 0.0

            x = self.last_layer.linear(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            out = x + 0.5
        else:
            # Shape (batch_size, num_points, dim_out)
            out = self.last_layer(x)
            # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        # Concatenate all parallel branches
        return out.view(*x_shape, out.shape[-1])


class ModulatedSirenResidual(SirenResidual):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        s0=0.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        # modulate_w0=False,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        last_layer_modulated=False,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            s0,
            w0_initial,
            use_bias,
        )

        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        # self.modulate_w0 = modulate_w0
        self.w0 = w0
        self.s0 = s0
        self.w0_initial = w0_initial
        self.last_layer_modulated = last_layer_modulated

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if last_layer_modulated:
            num_modulations += dim_out

        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        # if self.modulate_w0:
        #     num_modulations += num_layers - 1

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        print(self.__dict__)

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )
        idx = 0

        h = None

        for i_layer, module in enumerate(self.net, start=1):
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx : idx + self.dim_hidden].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[
                    :, mid_idx + idx : mid_idx + idx + self.dim_hidden
                ].unsqueeze(1)
            else:
                shift = 0.0

            if i_layer % 2:
                h = module.linear(x)
                h = scale * h + shift  # Broadcast scale and shift across num_points
                h = module.activation(h)  # (batch_size, num_points, dim_hidden)
            else:
                x = module.linear(h)
                x = scale * x + shift
                x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.dim_hidden

        if self.last_layer_modulated:
            if self.modulate_scale:
                scale = modulations[:, idx : idx + self.dim_out].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                shift = modulations[
                    :, mid_idx + idx : mid_idx + idx + self.dim_out].unsqueeze(1)
            else:
                shift = 0.0

            x = self.last_layer.linear(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            out = x + 0.5
        else:
            # Shape (batch_size, num_points, dim_out)
            out = self.last_layer(x)

        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenMultiBranch(SirenMultiBranch):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
            dim_in,
            dim_hidden_trunk,
            dim_hidden_branch,
            dim_out,
            num_layers_trunk,
            num_layers_branch,
            w0=30.0,
            s0=0.0,
            w0_initial=30.0,
            use_bias=True,
            modulate_scale=False,
            modulate_shift=True,
            # modulate_w0=False,
            use_latent=False,
            latent_dim=64,
            modulation_net_dim_hidden=64,
            modulation_net_num_layers=1,
            last_layer_modulated=False,
            modulate_trunk=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden_trunk,
            dim_hidden_branch,
            dim_out,
            num_layers_trunk,
            num_layers_branch,
            w0=w0,
            s0=s0,
            w0_initial=w0_initial,
            use_bias=use_bias,
        )
        self.dim_in = dim_in
        self.dim_hidden_branch = dim_hidden_branch
        self.dim_hidden_trunk = dim_hidden_trunk
        self.dim_out = dim_out
        self.num_layers_branch = num_layers_branch
        self.num_layers_trunk = num_layers_trunk
        self.w0 = w0
        self.s0 = s0
        self.w0_initial = w0_initial
        self.use_bias = use_bias

        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.modulate_trunk = modulate_trunk

        # self.modulate_w0 = modulate_w0
        self.w0 = w0
        self.s0 = s0
        self.w0_initial = w0_initial
        self.last_layer_modulated = last_layer_modulated

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden_branch * (num_layers_branch - 1) * dim_out
        if self.modulate_trunk:
            num_modulations += dim_hidden_trunk * (num_layers_trunk)

        if last_layer_modulated:
            num_modulations += dim_out

        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        # if self.modulate_w0:
        #     num_modulations += num_layers - 1

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        print(self.__dict__)
        # 1/0
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )

        idx = 0

        # print(f'modulations.shape: {modulations.shape}')

        for module in self.net_trunk:
            # print(f'idx: {idx}')
            if self.modulate_scale and self.modulate_trunk:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx : idx + self.dim_hidden_trunk].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift and self.modulate_trunk:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[
                    :, mid_idx + idx : mid_idx + idx + self.dim_hidden_trunk
                ].unsqueeze(1)
            else:
                shift = 0.0

            x = module.linear(x)
            # print(f'x.shape: {x.shape}')
            # print(scale.shape)
            # print(f'shift.shape: {shift.shape}')
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.dim_hidden_trunk

        out_trunk = x
        if not self.modulate_trunk:
            idx = 0

        for subnet in self.list_net_branch:
            x = out_trunk

            for module in subnet[:-1]:

                if self.modulate_scale:
                    # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                    # modulations remain zero centered
                    scale = modulations[:, idx: idx + self.dim_hidden_branch].unsqueeze(1) + 1.0
                else:
                    scale = 1.0

                if self.modulate_shift:
                    # Shape (batch_size, 1, dim_hidden)
                    shift = modulations[
                            :, mid_idx + idx: mid_idx + idx + self.dim_hidden_branch
                            ].unsqueeze(1)
                else:
                    shift = 0.0
                # print(f'x.shape: {x.shape}')
                # print(f'scale.shape: {scale.shape}')
                # print(f'shift.shape: {shift.shape}')
                # print(f'modulation.shape: {modulations.shape}')

                x = module.linear(x)
                x = scale * x + shift  # Broadcast scale and shift across num_points
                x = module.activation(x)  # (batch_size, num_points, dim_hidden)

                idx = idx + self.dim_hidden_branch

        out_branch = []

        if self.last_layer_modulated:
            for subnet in self.list_net_branch:
                last_layer = subnet[-1]

                if self.modulate_scale:
                    scale = modulations[:, idx : idx + 1].unsqueeze(1) + 1.0
                else:
                    scale = 1.0

                if self.modulate_shift:
                    shift = modulations[
                        :, mid_idx + idx : mid_idx + idx + 1].unsqueeze(1)
                else:
                    shift = 0.0

                x = last_layer.linear(x)
                x = scale * x + shift  # Broadcast scale and shift across num_points
                out = x + 0.5
                out_branch.append(out)
        else:
            for subnet in self.list_net_branch:
                last_layer = subnet[-1]
                # Shape (batch_size, num_points, dim_out)
                out_branch.append(last_layer(x))

        output = torch.cat(out_branch, dim=-1)
        # print(f'output.shape: {output.shape}')
        # 1/0
        return output

class LatentToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.

    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(self, latent_dim, num_modulations, dim_hidden, num_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), nn.ReLU()]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden), nn.ReLU()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)


class Bias(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size), requires_grad=True)
        # Add latent_dim attribute for compatibility with LatentToModulation model
        self.latent_dim = size

    def forward(self, x):
        return x + self.bias

class Empty(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 0
    def forward(self, x):
        return x


if __name__ == "__main__":
    dim_in, dim_hidden, dim_out, num_layers = 2, 5, 3, 4
    batch_size, latent_dim = 3, 7
    model = ModulatedSiren(
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        modulate_scale=True,
        use_latent=True,
        latent_dim=latent_dim,
    )
    print(model)
    latent = torch.rand(batch_size, latent_dim)
    x = torch.rand(batch_size, 5, 5, 2)
    out = model(x)
    out = model.modulated_forward(x, latent)
    print(out.shape)
