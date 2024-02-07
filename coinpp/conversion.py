import torch


class Converter:
    """Class that converts data to coordinates and features and back.

    Args:
        data_type (string): One of 'audio' or 'time-series'. The 'audio' type is mainly kept for backwards compatibility
        with the librespeech dataset.

    Notes:
        For images, MRI, ERA5 and audio we assume the coordinates are fixed so we don't
        recalculate them at every conversion.
    """

    def __init__(self, data_type="image"):
        assert data_type in ("audio", 'time-series')
        self.data_type = data_type
        self.coordinates = None

    def to_coordinates_and_features(self, data):
        """
        Args:
            data (torch.Tensor):
        """
        if self.data_type == "audio":
            # If first conversion, calculate coordinates, otherwise reuse
            if self.coordinates == None:
                # Data has shape ({batch_size,} channels, width)
                self.coordinates = shape2coordinates(data.shape[-1:]).to(data.device)
                # Scale data from [0, 1] to [-5, 5]
                self.coordinates = 10 * self.coordinates - 5
            # If data has 3 dimensions, it is batched
            if data.ndim == 3:
                coordinates = repeat_coordinates(self.coordinates, data.shape[0])
                features = data2features(data, batched=True)
            else:
                coordinates = self.coordinates
                features = data2features(data, batched=False)
            return coordinates, features

        elif self.data_type == "time-series":

            # If first conversion, calculate coordinates, otherwise reuse
            if self.coordinates == None:

                # Data has shape ({batch_size,} channels, width)
                self.coordinates = shape2coordinates(data.shape[-1:]).to(data.device)

            # If data has 3 dimensions, it is batched
            if data.ndim == 3:
                coordinates = repeat_coordinates(self.coordinates, data.shape[0])
                features = data2features(data, batched=True)
            else:
                coordinates = self.coordinates
                features = data2features(data, batched=False)

            return coordinates, features

        else:
            raise ValueError('data_type must be one of "audio", "time-series"')

    def to_data(self, coordinates, features):
        """
        Args:
            coordinates (torch.Tensor): Unused for 'era5', 'image', 'mri' and 'audio'.
            features (torch.Tensor):
        """
        return features2data(features, batched=features.ndim == 3)


def data2features(data: torch.Tensor, batched: bool = False) -> torch.Tensor:
    """Converts an audio sample, image or volume to a features tensor of shape
    ({batch,} {depth x height} x width}, channel).

    Args:
        data (torch.Tensor): Shape (batch_size, channels, *) if batched is True
            or (channels, *) if batched is False, where * refers to any spatial
            dimensions, e.g. (H, W).
        batched (bool): If True, considers first dimension as batch dimension.

    Returns:
        torch.Tensor: of shape (batch_size, *, channels) or (*, channels).
    """
    # Move channels dimension to last axis
    if batched:
        return torch.moveaxis(data, 1, -1)
    else:
        return torch.moveaxis(data, 0, -1)


def features2data(features, batched=False):
    """Inverse function of data2features."""
    # Move channels dimension to first non batch axis
    if batched:
        return torch.moveaxis(features, -1, 1)
    else:
        return torch.moveaxis(features, -1, 0)


def shape2coordinates(spatial_shape: torch.Size, batch_size: int = 0):
    """Converts a shape tuple to a tensor of coordinates.

    Args:
        spatial_shape (tuple of ints): Tuple describing shape of data. For
            example (height, width) or (depth, height, width).
        batch_size (int): If not zero, repeats the coordinate tensor to create
            a batch of coordinates.

    Notes:
        The coordinate tensor will have coordinates lying in [0, 1] regardless
        of the input shape. Be careful if you have inputs that have very non
        square shapes, e.g. (4, 128) as each coordinate grid might then need to
        be scaled differently.
    """
    coords = []
    for i in range(len(spatial_shape)):
        coords.append(torch.linspace(0.0, 1.0, spatial_shape[i]))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)


def repeat_coordinates(coordinates, batch_size):
    """Repeats the coordinate tensor to create a batch of coordinates.

    Args:
        coordinates (torch.Tensor): Shape (*spatial_shape, len(spatial_shape)).
        batch_size (int): If not zero, repeats the coordinate tensor to create
            a batch of coordinates.
    """
    if batch_size:
        ones_like_shape = (1,) * coordinates.ndim
        return coordinates.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    else:
        return coordinates


if __name__ == "__main__":

    converter = Converter("audio")
    audio = torch.rand(8, 1, 1000)
    coords, feats = converter.to_coordinates_and_features(audio)
    print(coords.shape)
    print(feats.shape)

    converter = Converter("time-series")
    ts = torch.rand(8, 1, 1000)
    coords, feats = converter.to_coordinates_and_features(ts)
    print(coords.shape)
    print(feats.shape)
