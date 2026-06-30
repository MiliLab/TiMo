import torch
import torch.nn as nn
from upernet_mmseg_30 import UPerHead

class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)

class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class SemsegFramework(torch.nn.Module):

    def __init__(self,
                 args,
                 classes: int = 1,
                 out_indices=[3,5,7,11]
                 ):
        super(SemsegFramework, self).__init__()

        self.args = args

        if args.model=='TiMo':
            import TiMo
            self.encoder = TiMo.__dict__[args.backbone] \
                (drop_path_rate=0.2, global_pool=True, img_size=args.img_size,
                 in_chans=args.in_chans)#drop_path_rate=0.2

        
        self.decoder = UPerHead(
            in_channels=[self.encoder.embed_dim for _ in range (4)] if args.model not in ['smt','TiMo','TiMo_spatRp','gfm','satlas','satlas_t','gfm_t','TiMo_monotp'] else self.encoder.embed_dim,
            channels=self.encoder.embed_dim if args.model not in ['smt','TiMo','TiMo_spatRp','gfm','satlas','satlas_t','gfm_t','TiMo_monotp'] else self.encoder.embed_dim[-1],
            in_index=(0, 1, 2, 3),
            dropout_ratio=0.1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.semseghead = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(self.encoder.embed_dim if args.model not in ['smt','TiMo','TiMo_spatRp','gfm','satlas','satlas_t','gfm_t','TiMo_monotp'] else self.encoder.embed_dim[-1], classes, kernel_size=1)
        )


        self.initialize()


    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.semseghead)

    def forward(self, x, ts):
        B,T,C,H,W=x.shape
        
        if self.args.model in ['TiMo']:
            features = self.encoder.forward_features(x)
        
        output = self.decoder(*features)
        output=output.view(B,T//self.args.tubelet_size,*output.shape[1:])
        if self.args.dataset not in ['MultiEarthDeforest']:
            output=torch.max(output,dim=1)[0]
            output = self.semseghead(output)
        else:
            output=self.semseghead(output.view(-1,*output.shape[2:]))
        
        return output
       

