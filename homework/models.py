from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        layer_dims = [64, 64],
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        # Define the MLP model
        l = [
            nn.Linear(n_track * 4, layer_dims[0]),
            nn.ReLU(),
        ]

        # Add hidden layers
        for d in layer_dims:
            l.append(nn.Linear(d, d))
            l.append(nn.ReLU())

        # Output layer
        l.append(nn.Linear(layer_dims[-1], n_waypoints * 2))

        # Save the number of points in each side of the track and the number of waypoints
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Concatenate all layers
        self.model = nn.Sequential(*l)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b_size, _, _ = track_left.size()

        # Flatten the input boundaries for the right side
        r_flat = track_right.view(b_size, -1)

        # Flatten the input boundaries for the left side
        l_flat = track_left.view(b_size, -1)

        # Concatenate the flattened boundaries
        c_input = torch.cat([l_flat, r_flat], dim=1)

        # Pass the concatenated input through the model
        out = self.model(c_input)

        # Reshape the output to the desired shape
        out = out.view(b_size, self.n_waypoints, 2)

        return out

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
    ):
        super().__init__()

       
        # Define the Transformer model
        e_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            nhead=heads,
            d_model=d_model,
            dim_feedforward=4 * d_model,
        )

        # Define the Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            e_layer,
            num_layers=layers
        )

        # Save the number of points in each side of the track and the number of waypoints
        self.n_track = n_track

        # Save the number of waypoints to predict and the input dimension
        self.n_waypoints = n_waypoints

        # Input projection to desired input dimension (d_model)
        self.input_projection = nn.Linear(4, d_model)
        
        # Output projection to desired output dimension (n_waypoints * 2)
        self.output_projection = nn.Linear(d_model, n_waypoints * 2)

    def forward(
        self,
        t_left: torch.Tensor,
        t_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Repeat flatenning similar to MLP
        b_size, _, _ = t_left.size()

        # Concatenate the left and right boundaries
        c_input = torch.cat([t_left, t_right], dim=-1) 
        
        # Pass the concatenated input through the input projection
        inp_embed = self.input_projection(c_input)  

        # Pass the input embedding through the transformer encoder
        enc_out = self.transformer_encoder(inp_embed)  
        
        # Average the encoder output
        pld_output = enc_out.mean(dim=1)  

        # Pass the encoder
        output = self.output_projection(pld_output)  
        
        # Reshape the output to the desired shape (b, n_waypoints, 2)
        return output.view(b_size, self.n_waypoints, 2)


class CNNPlanner(torch.nn.Module):
    

    class DecoderBlock(nn.Module):
        def __init__(
            self,
            in_c,
            out_c,
        ):
            super().__init__()
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c,  stride=2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(out_c, out_c,  stride=1, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.conv(x)
        
    class EncoderBlock(nn.Module):
        def __init__(
            self,
            in_c,
            out_c,
        ):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, stride=2, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Dropout2d(0.2),
                nn.Conv2d(out_c, out_c, stride=1, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Dropout2d(0.2),
            )

        def forward(self, x):
            return self.conv(x)
    def __init__(
        self,
        n_wpts: int = 3,
        in_c: int = 3,
        feats = [32, 64, 128],
    ):
        super().__init__()

        self.n_waypoints = n_wpts

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Define the CNN model (Encoder-Decoder)
        dec_l = []

        enc_l = [
            nn.Conv2d(in_c, feats[0], kernel_size=3, stride=1, padding=1)
        ]

        for f in feats:
            enc_l.append(self.EncoderBlock(f, f * 2))

        for f in reversed(feats):
            dec_l.append(self.DecoderBlock(f * 2, f))

        # Output layer
        self.encoder = nn.Sequential(*enc_l)
        self.decoder = nn.Sequential(*dec_l)

        # Set up the head (output layer)
        self.head = nn.Conv2d(feats[0], n_wpts * 2, kernel_size=1)
        
        # Set up Global Pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        img_tensor = image
        # Normalize the input image
        img_tensor = (img_tensor - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        # Pass the input image through the encoder
        img_tensor = self.encoder(img_tensor)
        # Pass the encoder output through the decoder
        img_tensor = self.decoder(img_tensor)
        # Pass the decoder output through the head
        img_tensor = self.head(img_tensor)
        # Calculate the global average pooling
        img_tensor = self.global_pool(img_tensor)
        # Reshape the output to the desired shape (b, n_waypoints, 2)
        img_tensor = img_tensor.view(img_tensor.size(0), self.n_waypoints, 2)

        # Return the output tensor with the desired shape (b, n_waypoints, 2)
        return img_tensor


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
