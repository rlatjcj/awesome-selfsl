
from typing import Optional, Tuple, Any

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from . import resnets



###############################
########### PyTorch ###########
###############################
"""
"""

class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int = 2048, 
        hidden_size: int = 4096, 
        output_dim: int = 256
    ) -> None:

        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SiameseArm(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        input_dim: int = 2048,
        hidden_size: int = 4096,
        output_dim: int = 256
    ) -> None:

        super().__init__()

        def torchvision_ssl_encoder(
            name: str,
            pretrained: bool = False,
            return_all_feature_maps: bool = False,
        ) -> nn.Module:
            pretrained_model = getattr(resnets, name)(
                pretrained=pretrained, 
                return_all_feature_maps=return_all_feature_maps)
            pretrained_model.fc = Identity()
            return pretrained_model

        if encoder is None:
            encoder = torchvision_ssl_encoder("resnet50")

        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(input_dim, hidden_size, output_dim)
        # Predictor
        self.predictor = MLP(output_dim, hidden_size, output_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        y = self.encoder(x)[0]
        z = self.projector(y)
        h = self.predictor(z)

        return y, z, h
        

class SimSiamPT(pl.LightningModule):

    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
        num_nodes: int = 1,
        arch: str = "resnet50",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = "adam",
        lars_wrapper: bool = True,
        exclude_bn_bias: bool = False,
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        weight_decay: float = 1e-6,
        **kwargs: Any
    ) -> None:

        super().__init__()
        
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.lars_wrapper = lars_wrapper
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.init_model()

    def init_model(self) -> None:
        if self.arch in ["resnet18", "resnet50"]:
            backbone = getattr(resnets, self.arch)
            encoder = backbone(
                first_conv=self.first_conv, 
                maxpool1=self.maxpool1, 
                return_all_feature_maps=False)
        else:
            encoder = None
            
        self.online_network = SiameseArm(
            encoder, 
            input_dim=self.hidden_mlp, 
            hidden_size=self.hidden_mlp, 
            output_dim=self.feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _, _ = self.online_network(x)
        return y

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        b = b.detach() # stop gradient of backbone + projection mlp
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = -1 * (a * b).sum(-1).mean()
        return sim

    def training_step(self, batch, batch_dix):
        (img_1, img_2, _), y = batch

        # img_1 to img_2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) + self.cosine_similarity(h2, z1)
        loss /= 2

        # log results
        self.log_dict({"loss": loss})

        return loss

    def validation_step(self, batch, batch_dix):
        (img_1, img_2, _), y = batch

        # img_1 to img_2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) + self.cosine_similarity(h2, z1)
        loss /= 2

        # log results
        self.log_dict({"loss": loss})

        return loss