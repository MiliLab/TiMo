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

from einops import rearrange
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


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class Attention(nn.Module):
    def __init__(self, dim, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                       attn_drop=0., proj_drop=0., local=1):
        super().__init__()

        self.local = local
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
        self.INF=INF
        if self.local==2:
            self.cheap_operation = nn.Sequential(  
                nn.Linear(head_dim, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.ReLU(inplace=True),
            )
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

    def forward(self, x, numUnit):
        B, N, C = x.shape
        H = W = int((N // numUnit) ** 0.5)

        if self.local==0:
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        elif self.local==1:
            T=numUnit
            N_=H*W
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).view(B,T,N_,self.sa_num_heads, C // self.sa_num_heads).permute(0, 3, 1,2, 4)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4).view(2,B,self.sa_num_heads,T,N_,C // self.sa_num_heads)
            k, v = kv[0], kv[1]#B,h,T,NTk,C


            proj_query_T = rearrange(q,"B h T N C->(B h N) T C")
            proj_query_N = rearrange(q,"B h T N C->(B h T) N C")

            proj_key_T = rearrange(k, "B h T N C->(B h N) C T")
            proj_key_N = rearrange(k, "B h T N C->(B h T) C N")


            proj_value_T = rearrange(v, "B h T N C->(B h N) T C")
            proj_value_N = rearrange(v,"B h T N C->(B h T) N C")
            energy_T = (torch.matmul(proj_query_T, proj_key_T) + self.INF(B*self.sa_num_heads, T, N_)).view(B*self.sa_num_heads,
                                                                                                     N_, T,
                                                                                                     T).permute(0, 2, 1,
                                                                                                                3)
            energy_N = torch.matmul(proj_query_N, proj_key_N).view(B*self.sa_num_heads, T, N_, N_)
            concate = self.softmax(torch.cat([energy_T, energy_N], 3))
            concate = self.attn_drop(concate)

            att_T = concate[:, :, :, 0:T].permute(0, 2, 1, 3).contiguous().view(B *self.sa_num_heads* N_, T, T)
            att_N = concate[:, :, :, T:T + N_].contiguous().view(B *self.sa_num_heads * T, N_, N_)

            out_T = torch.matmul(att_T, proj_value_T).view(B *self.sa_num_heads, N_, T, C// self.sa_num_heads).permute(0, 2, 1, 3)
            out_N = torch.matmul(att_N, proj_value_N).view(B *self.sa_num_heads, T, N_, C// self.sa_num_heads)

            output = out_T + out_N
            x = rearrange(output, "(B h) T N C->B (T N) (h C)", B=B)

        elif self.local==2:
            T = numUnit
            N_ = H * W
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).view(B, T, N_, self.sa_num_heads,
                                                                                        C // self.sa_num_heads).permute(
                0, 3, 1, 2, 4)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4).view(2,
                                                                                                                     B,
                                                                                                                     self.sa_num_heads,
                                                                                                                     T,
                                                                                                                     N_,
                                                                                                                     C // self.sa_num_heads)
            k, v = kv[0], kv[1]  # B,h,T,NTk,C

            proj_query_T = rearrange(q, "B h T N C->(B h N) T C")
            proj_query_N = rearrange(q, "B h T N C->(B h) T N C")

            proj_key_T = rearrange(k, "B h T N C->(B h N) C T")
            proj_key_N = rearrange(k, "B h T N C->(B h) T C N")

            proj_value_T = rearrange(v, "B h T N C->(B h N) T C")
            proj_value_N = rearrange(v, "B h T N C->(B h T) N C")
            energy_T = (torch.matmul(proj_query_T, proj_key_T) + self.INF(B * self.sa_num_heads, T, N_)).view(
                B * self.sa_num_heads,
                N_, T,
                T).permute(0, 2, 1,
                           3)
            energy_N = torch.matmul(torch.median(proj_query_N, dim=1)[0], torch.median(proj_key_N, dim=1)[0]).unsqueeze(
                1).repeat(1, T, 1, 1)
            energy_N_diff = proj_query_N - torch.median(proj_query_N, dim=1)[0].unsqueeze(1).repeat(1, T, 1,
                                                                                                    1)  # (B h) T N C
            energy_N_diff = energy_N_diff.contiguous().view(-1, energy_N_diff.shape[-1])

            energy_N_diff = self.cheap_operation(energy_N_diff)  # (B h) T N 1
            energy_N_diff = energy_N_diff.view(energy_N.shape[0], T, N_, 1).repeat(1, 1, 1, N_)
            energy_N = energy_N + energy_N_diff
            concate = self.softmax(torch.cat([energy_T, energy_N], 3))
            concate = self.attn_drop(concate)

            att_T = concate[:, :, :, 0:T].permute(0, 2, 1, 3).contiguous().view(B * self.sa_num_heads * N_, T, T)
            att_N = concate[:, :, :, T:T + N_].contiguous().view(B * self.sa_num_heads * T, N_, N_)

            out_T = torch.matmul(att_T, proj_value_T).view(B * self.sa_num_heads, N_, T,
                                                           C // self.sa_num_heads).permute(0, 2, 1, 3)
            out_N = torch.matmul(att_N, proj_value_N).view(B * self.sa_num_heads, T, N_, C // self.sa_num_heads)


            output = out_T + out_N
            x = rearrange(output, "(B h) T N C->B (T N) (h C)", B=B)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim,  sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                    use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, local=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, local=local
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

    def forward(self, x,numUnit):

        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x),numUnit))#b,n,c
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Head(nn.Module):
    def __init__(self, in_chans,head_conv, dim):
        super(Head, self).__init__()
        stem = [nn.Conv2d(in_chans, dim, head_conv, 2, padding=3 if head_conv==7 else 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True)]
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

        return x

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
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x



class TiMo(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[8, 6, 4, 2],
                 qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 8, 1], local=[1, 1, 1, 0], num_stages=4, head_conv=7, use_pos=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dim=embed_dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i ==0:
                patch_embed = Head(in_chans,head_conv, embed_dims[i])#PatchEmbed(img_size, patch_size, in_chans, embed_dims[i])
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=3,
                                            stride=2,
                                            in_chans=embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], sa_num_heads=sa_num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                use_layerscale=use_layerscale, 
                layerscale_value=layerscale_value,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                local=local[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

       
        self.use_pos=use_pos

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

    def forward_features(self, x):

        B = x.shape[0]
        T=x.shape[1]
        if len(x.shape) == 5:
            x = x.view(-1, *x.shape[2:])

        output = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x = patch_embed(x)#[4, 3136, 64]->[4, 784, 128]->[4, 196, 256]->[4, 49, 256]
            numUnit=T
            x=x.view(B,T,x.shape[1],x.shape[-1]).contiguous().view(B,T*x.shape[1],x.shape[-1])

            # --------consider positional encoding------------
            if self.use_pos == 'ts' and i==0:
                from pos_embed import get_1d_sincos_pos_embed_from_grid_torch, get_2d_sincos_pos_embed
                timestamps = torch.tensor(range(T))
                t_embed = get_1d_sincos_pos_embed_from_grid_torch(self.embed_dim[0] // 2,
                                                                  timestamps.float()).unsqueeze(0).unsqueeze(2)
                t_embed = t_embed.expand(x.shape[0], -1, x.shape[1] // T, -1).contiguous().view(x.shape[0], -1,
                                                                                                t_embed.shape[-1])
                s_embed = get_2d_sincos_pos_embed(self.embed_dim[0] // 2, int((x.shape[1] // T) ** .5),
                                                  cls_token=False)
                s_embed = torch.from_numpy(s_embed).float().unsqueeze(0).repeat(t_embed.shape[0], T, 1)
                ts_embed = torch.concat([t_embed, s_embed], dim=-1).to(x.device)
                x += ts_embed

            elif self.use_pos == 's' and i==0:
                from pos_embed import get_2d_sincos_pos_embed
                s_embed = get_2d_sincos_pos_embed(self.embed_dim[0], int((x.shape[1] // T) ** .5), cls_token=False)
                s_embed = torch.from_numpy(s_embed).float().unsqueeze(0).repeat(x.shape[0], T, 1)
                ts_embed = s_embed.to(x.device)
                x += ts_embed

            for blk in block:
                #the input x.shape is [B,T*num_Unit*num_tokensPerUnit,C]
                x = blk(x, numUnit)
            x = norm(x)
            
            x_intermediate = x.view(x.shape[0] * T, x.shape[1] // T, x.shape[-1])  # [B*T,numUnit_perFrame*num_tokensperUnit,C]
            x_intermediate = x_intermediate.view(x_intermediate.shape[0], int(x_intermediate.shape[1] ** 0.5),
                       int(x_intermediate.shape[1] ** 0.5), x_intermediate.shape[-1]).permute(0,3,1,2)

            output.append(x_intermediate)
            if i != self.num_stages - 1:
                x = x.reshape(B*numUnit,int((x.shape[1]//numUnit)**0.5),int((x.shape[1]//numUnit)**0.5), -1).permute(0, 3, 1, 2).contiguous()

        return output

    def forward(self, x,ts):
        x = self.forward_features(x)
        
        return x



def TiMo_base(pretrained=False, **kwargs):
    model = TiMo(
        embed_dims=[128, 256,512, 1024],  sa_num_heads=[4,8,16,32], mlp_ratios=[4,4,4,4],
        qkv_bias=True, depths=[2, 2, 18, 2], local=[2, 2, 0, 0], head_conv=7,use_pos='ts', **kwargs)
    model.default_cfg = _cfg()

    return model


def TiMo_large(**kwargs):
    model = TiMo(
        embed_dims=[384, 768, 960, 1536], sa_num_heads=[ 6, 12, 24, 48 ], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, depths=[2, 2, 18, 2], local=[2, 2, 0, 0], head_conv=7,
        use_pos='ts', **kwargs)
    model.default_cfg = _cfg()

    return model


def TiMo_huge(**kwargs):
    model = TiMo(
        embed_dims=[352,704,1408,2816], sa_num_heads=[ 8, 16, 32, 64 ], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, depths=[2, 2, 18, 2], local=[2, 2, 0, 0], head_conv=7,
        use_pos='ts', **kwargs)
    model.default_cfg = _cfg()

    return model

