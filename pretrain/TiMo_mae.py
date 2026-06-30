import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp

from timm.models.vision_transformer import PatchEmbed



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim

        self.sa_num_heads = sa_num_heads

        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        head_dim = dim // sa_num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.softmax = nn.Softmax(dim=3)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))  # b,n,c
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Head(nn.Module):
    def __init__(self, head_conv, dim):
        super(Head, self).__init__()
        stem = [nn.Conv2d(3, dim, head_conv, 2, padding=3 if head_conv==7 else 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True)]
        stem.append(nn.Conv2d(dim, dim, kernel_size=2, stride=2))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x = self.conv(x)
       
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x#, H, W

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        img_size = to_2tuple(img_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x = self.proj(x)  # 下采样两倍
       
        _, _, H, W = x.shape
        
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
       
        return x#, H, W

def get_resized_mask(target_size: torch.Size, mask: torch.Tensor) -> torch.Tensor:
    # target_size: [(T), (H), W]
    # (spatial) mask: [B, C, (t), (h), w]
    if mask is None:
        return mask

    assert len(mask.shape[2:]) == len(target_size)

    if mask.shape[2:] != target_size:
        return F.interpolate(mask.float(), size=target_size)

    return mask

class MaskedAutoencoderTiMo(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[128, 256, 512, 1024],
                 sa_num_heads=[4, 8, 16, 32], mlp_ratios=[4,4,4,4],
                 qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 18, 2],  num_stages=4, head_conv=7,
                 decoder_embed_dim=512, decoder_mlp_ratios=4, decoder_depth=8, use_pos=None, decoder_heads=16,
                 norm_pix_loss=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.norm_pix_loss = norm_pix_loss

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed =  Head(head_conv, embed_dims[i])#PatchEmbed(img_size, 4, in_chans, embed_dims[i])
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                                patch_size=3,
                                                stride=2,
                                                in_chans=embed_dims[i - 1],
                                                embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], sa_num_heads=sa_num_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                )
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dims[-1], decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim, sa_num_heads=decoder_heads, mlp_ratio=decoder_mlp_ratios,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    use_layerscale=use_layerscale,
                    layerscale_value=layerscale_value,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                    )  # 0 if i==2 and j%2!=0 else 1
                for j in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            ((4 * 8) ** 2) * in_chans
        )  # predictor

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.use_pos = use_pos
        if use_pos is not None:
            self.embed_dims = embed_dims
            self.decoder_embed_dim = decoder_embed_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def get_random_mask(self, B,num_windows,device, mask_ratio):
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """

        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(B, num_windows, device=device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([B, num_windows], device=device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool(), ids_restore



    def get_pixel_label_2d(
            self, input_img: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*
        if len(input_img.shape) < 5:
            input_img = input_img.permute(0, 2, 3, 1)

            size = int(input_img.shape[1] // (mask.shape[1] ** 0.5))
            label = input_img.unfold(1, size, size).unfold(2, size, size)
            
            label = label.flatten(1, 2).flatten(2)
            
            label = label.contiguous().view(*label.shape[:3], -1)
            mask = mask.repeat(1, 1, 1, label.shape[-1])
            label = label[mask]
        elif len(input_img.shape) == 5:
            B, T, C, H, W = input_img.shape
            input_img = input_img.view(-1, *input_img.shape[2:])
            input_img = input_img.permute(0, 2, 3, 1)
            size = input_img.shape[1] // (mask.shape[1])
            label = input_img.unfold(1, size, size).unfold(2, size, size)
            
            label = label.contiguous().view(*label.shape[:3], -1)
            mask = mask.repeat(1, 1, 1, label.shape[-1])

            label = label[mask]
        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def forward_encoder(self, x, mask_ratio):
        B = x.shape[0]
        T = 1
        ts = False
        if len(x.shape) == 5:
            ts = True
            T = x.shape[1]
            x = x.view(-1, *x.shape[2:])
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            if i==0:
                mask, ids_restore = self.get_random_mask(B,(x.shape[-1]//32)**2*T,x.device, mask_ratio=mask_ratio)  # B,num_Unit
                
                mask = mask.view(mask.shape[0],T,x.shape[-1]//32,x.shape[-1]//32).unsqueeze(1).repeat(1, x.shape[1],1,1,  1).bool()#B,C,T,7,7

                mask_resized=get_resized_mask(target_size=[3,224,224],mask=mask)
                mask_resized=mask_resized.transpose(1,2).contiguous().view(B*T,x.shape[1],*mask_resized.shape[-2:])#B*T,C,224,224

                x=x*mask_resized

            x= patch_embed(x)  # [4, 3136, 64]->[4, 784, 128]->[4, 196, 256]->[4, 49, 256]

            if self.use_pos == 'ts' and i == 0:
                from pos_embed import get_1d_sincos_pos_embed_from_grid_torch, get_2d_sincos_pos_embed
                # x.shape->[6,3136,C]
                timestamps = torch.tensor([range(T) for _ in range(B)])
                t_embed = get_1d_sincos_pos_embed_from_grid_torch(self.embed_dims[0] // 2,
                                                                  timestamps.reshape(-1).float()).unsqueeze(1)
                t_embed = t_embed.expand(-1, x.shape[1], -1)

                s_embed = get_2d_sincos_pos_embed(self.embed_dims[0] // 2, int(x.shape[1] ** .5), cls_token=False)
                s_embed = torch.from_numpy(s_embed).float().unsqueeze(0).repeat(t_embed.shape[0], 1, 1)
                
                ts_embed = torch.concat([t_embed, s_embed], dim=-1).to(x.device)
                x += ts_embed


            elif self.use_pos == 's' and i == 0:
                from pos_embed import get_2d_sincos_pos_embed
                s_embed = get_2d_sincos_pos_embed(self.embed_dims[0], int(x.shape[1] ** .5), cls_token=False)
                s_embed = torch.from_numpy(s_embed).float().unsqueeze(0).repeat(B * T, 1, 1)
                ts_embed = s_embed.to(x.device)
                x += ts_embed

            if i == 0:
                x = x.permute(0, 2, 1).contiguous().view(x.shape[0], x.shape[-1], int(x.shape[1] ** 0.5),
                                                         int(x.shape[1] ** 0.5))
                unfold = torch.nn.Unfold(kernel_size=(8, 8), stride=8)
                x_unfold = unfold(x)
                
                x_unfold = x_unfold.view(x_unfold.shape[0], x.shape[1], -1,
                                         x_unfold.shape[-1])  # B*T,C,num_tokens_perUnit,num_Unit
                

                if ts == True:
                    x_unfold = x_unfold.view(B, T, *x_unfold.shape[1:]).permute(0, 2, 3, 1, 4).contiguous().view(B,
                                                                                                          x_unfold.shape[
                                                                                                              1],
                                                                                                          x_unfold.shape[
                                                                                                              2], T *
                                                                                                          x_unfold.shape[
                                                                                                              3])
                    
                mask = mask[:,0].view(mask.shape[0],T*(mask.shape[-1]**2))
                mask=mask.unsqueeze(1).unsqueeze(2).repeat(1,x_unfold.shape[1],x_unfold.shape[2],1)#B,C,numTperU,T*numU

                x_unfold = x_unfold[mask].view(*x_unfold.shape[:-1], -1)  # B,C,num_tokens_perUnit,num_Unit'

                numUnit = x_unfold.shape[-1]

                x = x_unfold.permute(0, 3, 2, 1).contiguous().view(x_unfold.shape[0], -1, x_unfold.shape[1])# B,num_Unit'*num_tokens_perUnit,C

            else:
                x = x.view(B, -1, x.shape[-1])
            for blk in block:
                x = blk(x)

            x = norm(x)
            if i != self.num_stages - 1:

                x = x.reshape(B * numUnit, int((x.shape[1] // numUnit) ** 0.5), int((x.shape[1] // numUnit) ** 0.5),
                              -1).permute(0, 3, 1, 2).contiguous()


        return x, mask, ids_restore, numUnit, T

    def forward_decoder(self, x, mask, numUnit, T):

        # Embed tokens
        x = self.decoder_embed(x)
        x = x.view(x.shape[0], numUnit, int((x.shape[1] // numUnit) ** 0.5), int((x.shape[1] // numUnit) ** 0.5),
                   x.shape[-1])  # [B, num_Unit',1,1, D]


        # 参考hiera_mae
        mask = mask[:, 0, 0, :]
        numUnit_ori = mask.shape[1]
        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)
        mask_tokens = self.mask_token.view(
            (1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,)
        )

        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        
        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        
        x_dec[mask] = x.flatten()
        x = ~mask * mask_tokens + mask * x_dec

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if self.use_pos == 'ts':
            from pos_embed import get_1d_sincos_pos_embed_from_grid_torch, get_2d_sincos_pos_embed
            timestamps = torch.tensor(range(T))
            
            t_embed = get_1d_sincos_pos_embed_from_grid_torch(self.decoder_embed_dim // 2,
                                                              timestamps.float()).unsqueeze(0)
            
            t_embed = t_embed.reshape(1, T, t_embed.shape[-1]).unsqueeze(2)

            t_embed = t_embed.expand(x.shape[0], -1, x.shape[1] // 3, -1).reshape(x.shape[0], -1, t_embed.shape[-1])

            s_embed = get_2d_sincos_pos_embed(self.decoder_embed_dim // 2, int((x.shape[1] // 3) ** .5),
                                              cls_token=False)

            s_embed = torch.from_numpy(s_embed).float().unsqueeze(0).repeat(t_embed.shape[0], 3, 1)
            ts_embed = torch.concat([t_embed, s_embed], dim=-1).to(x.device)
            x += ts_embed

        elif self.use_pos == 's':
            from pos_embed import get_2d_sincos_pos_embed
            s_embed = get_2d_sincos_pos_embed(self.decoder_embed_dim, int((x.shape[1] // 3) ** .5), cls_token=False)
            s_embed = torch.from_numpy(s_embed).float().unsqueeze(0).repeat(x.shape[0], 3, 1)
            ts_embed = s_embed.to(x.device)
            x += ts_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        shape1 = (int(((mask.shape[1] / T) ** 0.5) * mask.shape[2]), int(((mask.shape[1] / T) ** 0.5) * mask.shape[2]))
        shape2 = (mask.shape[2], mask.shape[3])

        x = x.view(x.shape[0] * T, x.shape[1] // T, x.shape[-1])  # [B*T,numUnit_perFrame*num_tokensperUnit,C]
        x = x.view(x.shape[0], mask.shape[1] // T, int((x.shape[1] / (mask.shape[1] // T)) ** 0.5),
                   int((x.shape[1] / (mask.shape[1] // T)) ** 0.5), x.shape[-1])
        mask = mask.view(mask.shape[0], T, mask.shape[1] // T, *mask.shape[2:]).contiguous().view(-1,
                                                                                                  mask.shape[1] // T,
                                                                                                  *mask.shape[2:])

        x = undo_windowing(x, shape1, shape2)
        mask = undo_windowing(mask[..., 0:1], shape1, shape2)

        return x, mask

    def forward_loss(
            self, x, pred, mask
    ):
        """
        Note: in mask, 0 is *visible*, 1 is *masked*

        x: e.g. [B, 3, H, W]
        pred: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        label: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        """

        label = self.get_pixel_label_2d(x, mask, norm=self.norm_pix_loss)
        
        pred = pred[mask.repeat(1, 1, 1, pred.shape[-1])]

        loss = (pred - label) ** 2


        return loss.mean(), pred, label

    def forward(self, x, dates, mask_ratio=0.75):
        latent, mask, ids_restore, numUnit, T = self.forward_encoder(x, mask_ratio)
        pred, pred_mask = self.forward_decoder(latent, mask, numUnit, T)

        return self.forward_loss(x, pred, ~pred_mask)


def undo_windowing(
        x: torch.Tensor, shape, mu_shape
) -> torch.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.

    Args:
        x: organized by mask units windows, e.g. in 2d [B, #MUy*#MUx, MUy, MUx, C]
        shape: current spatial shape, if it were not organized into mask unit
            windows, e.g. in 2d [B, #MUy*MUy, #MUx*MUx, C].
        mu_shape: current mask unit shape, e.g. in 2d [MUy, MUx]
    Returns:
        x: e.g. in 2d, [B, #MUy*MUy, #MUx*MUx, C]
    """
    D = len(shape)
    B, C = x.shape[0], x.shape[-1]
    # [B, #MUy*#MUx, MUy, MUx, C] -> [B, #MUy, #MUx, MUy, MUx, C]
    num_MUs = [s // mu for s, mu in zip(shape, mu_shape)]
    x = x.view(B, *num_MUs, *mu_shape, C)

    # [B, #MUy, #MUx, MUy, MUx, C] -> [B, #MUy*MUy, #MUx*MUx, C]
    permute = (
            [0]
            + sum(
        [list(p) for p in zip(range(1, 1 + D), range(1 + D, 1 + 2 * D))],
        [],
    )
            + [len(x.shape) - 1]
    )
    x = x.permute(permute).reshape(B, *shape, C)

    return x


def TiMo_base(**kwargs):
    model = MaskedAutoencoderTiMo(
        embed_dims=[128, 256, 512, 1024], sa_num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, depths=[2, 2, 18, 2], head_conv=7,  decoder_embed_dim=512, use_pos='ts',
        **kwargs)
    model.default_cfg = _cfg()

    return model

def TiMo_large(**kwargs):
    model = MaskedAutoencoderTiMo(
        embed_dims=[384, 768, 960, 1536], sa_num_heads=[ 6, 12, 24, 48 ], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, depths=[2, 2, 18, 2],  head_conv=7,  decoder_embed_dim=512,
        use_pos='ts', **kwargs)
    model.default_cfg = _cfg()

    return model


def TiMo_huge(**kwargs):
    model = MaskedAutoencoderTiMo(
        embed_dims=[352,704,1408,2816], sa_num_heads=[ 8, 16, 32, 64 ], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, depths=[2, 2, 18, 2], head_conv=7,decoder_embed_dim=512,
        use_pos='ts', **kwargs)
    model.default_cfg = _cfg()

    return model
