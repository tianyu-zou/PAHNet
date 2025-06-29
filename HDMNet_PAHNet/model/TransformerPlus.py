import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer, ConvModule
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import constant_init, normal_init,trunc_normal_init
from mmcv.runner import BaseModule, ModuleList, Sequential
from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from mmseg.ops import resize
from HDMNet_PAHNet.model.MaskMultiheadAttention import MaskMultiHeadAttention


class MixFFN(BaseModule):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = MaskMultiHeadAttention(
            in_features=embed_dims, head_num=num_heads, bias=False, activation=None
        )
        torch.nn.MultiheadAttention

    def forward(self, x, hw_shape, source=None, identity=None, mask=None, cross=False, pro_pred=None):
        x_q = x
        if source is None:
            x_kv = x
        else:
            x_kv = source

        mask_ver = None
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x_kv, hw_shape)
            x_kv = self.sr(x_kv)
            size = x_kv.size()[-2:]
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)

            if mask is not None:
                # mask: (4, 5*60*60, 1)
                # x_kv: (20, 15*15, 64)
                if mask.size(1) != x_kv.size(1):
                    if mask.size(0) != x_kv.size(0):
                        shot = x_kv.size(0) // mask.size(0)
                        mask = rearrange(mask, 'b (n l) c -> (b n) l c', b=mask.size(0), n=shot)  # (bs*shot, h*w, 1)
                    mask_ver = nlc_to_nchw(mask, hw_shape)  # (bs*shot, 1, h, w)
                    mask_ver = F.interpolate(mask_ver, size=size, mode='bilinear', align_corners=True)
                    mask_ver = nchw_to_nlc(mask_ver)  # (bs*shot, h*w, c)

        if identity is None:
            identity = x_q

        out, weight = self.attn(q=x_q, k=x_kv, v=x_kv, mask=mask, cross=cross, mask_ver=mask_ver, pro_pred=pro_pred)
        return identity + self.dropout_layer(self.proj_drop(out)), weight


class TransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1):
        super(TransformerEncoderLayer, self).__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    def forward(self, x, hw_shape, source=None, mask=None, cross=False, pro_pred=None):
        if source is None:
            # Self attention
            x, weight = self.attn(self.norm1(x), hw_shape, identity=x, mask=mask)
        else:
            # Cross attention
            x, weight = self.attn(self.norm1(x), hw_shape, source=self.norm1(source), identity=x, mask=mask, cross=cross, pro_pred=pro_pred)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x, weight


class Mlp(nn.Module):
    """ Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
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


class PrototypeEnhanced(nn.Module):
    def __init__(self, embed_dims, drop=0., mlp_ratio=4):
        super(PrototypeEnhanced, self).__init__()
        self.embed_dims = embed_dims
        self.drop = drop
        self.norm1 = nn.LayerNorm(embed_dims)

        self.q = nn.Linear(embed_dims, embed_dims, bias=False)
        self.k = nn.Linear(embed_dims, embed_dims, bias=False)
        self.v = nn.Linear(embed_dims, embed_dims, bias=False)

        self.proj_x1 = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        self.proj_y1 = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        self.proj_x2 = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        self.proj_y2 = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        self.final_proj_x = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        self.final_proj_y = nn.Linear(embed_dims * 2, embed_dims, bias=False)

        self.norm2 = nn.LayerNorm(embed_dims)

        self.ffn_x = Mlp(in_features=embed_dims, hidden_features=embed_dims * mlp_ratio, act_layer=nn.GELU, drop=drop)
        self.ffn_y = Mlp(in_features=embed_dims, hidden_features=embed_dims * mlp_ratio, act_layer=nn.GELU, drop=drop)

    def Weighted_GAP(self, supp_feat, mask):
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat

    def forward(self, x, y, shape, mask, pro_pred):
        b, _, c = x.size()
        h, w = shape

        # skip connection
        x_skip = x  # b, n, c
        y_skip = y  # b, n, c

        # reshape
        mask = mask.view(b, -1, w, 1).permute(0, 3, 1, 2).contiguous()  # b, 1, shot*h, w
        # layer norm
        x = self.norm1(x)
        y = self.norm1(y)

        # input projection
        q = self.q(y)  # Support: b, n, c
        k = self.k(x)  # Query: b, n, c
        v = self.v(x)  # Query: b, n, c

        # query prototype extraction
        feature_s = q.view(b, -1, w, c).permute(0, 3, 1, 2).contiguous()  # b, c, h, w
        feature_q = k.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # b, c, h, w
        feature_qfp = v.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # b, c, h, w

        # support prototype extraction
        fg_pro = self.Weighted_GAP(feature_s, mask).squeeze(-1)  # b, c, 1
        bg_pro = self.Weighted_GAP(feature_s, 1 - mask).squeeze(-1)  # b, c, 1

        # calculate query FG/BG cosine 相似度
        similarity_fg = F.cosine_similarity(feature_q, fg_pro.unsqueeze(-1), dim=1) * 10.0
        similarity_bg = F.cosine_similarity(feature_q, bg_pro.unsqueeze(-1), dim=1) * 10.0

        # softmax
        pseudo_mask = torch.softmax(torch.concat([similarity_bg.unsqueeze(1), similarity_fg.unsqueeze(1)], 1), 1)

        # Affinity-based prototype fusing
        query_fg1 = self.Weighted_GAP(feature_qfp, pseudo_mask[:, 1:2, :, :]).squeeze(-1)
        sim1 = F.cosine_similarity(query_fg1, fg_pro, dim=1).unsqueeze(-1)
        sim1 = (sim1 + 1.) / 2.  # b, 1, 1
        pro1 = sim1 * fg_pro + (1. - sim1) * query_fg1  # b, c, 1

        # Prototype-based prototype fusing
        query_fg2 = self.Weighted_GAP(feature_qfp, pro_pred).squeeze(-1)
        sim2 = F.cosine_similarity(query_fg2, fg_pro, dim=1).unsqueeze(-1)
        sim2 = (sim2 + 1.) / 2.  # b, 1, 1
        pro2 = sim2 * fg_pro + (1. - sim2) * query_fg2  # b, c, 1

        # Feature enhance
        x1 = self.proj_x1(torch.cat([x_skip, pro1.permute(0, 2, 1).expand_as(x_skip)], dim=-1))
        x2 = self.proj_x2(torch.cat([x_skip, pro2.permute(0, 2, 1).expand_as(x_skip)], dim=-1))
        x_merge = self.final_proj_x(torch.cat([x1, x2], dim=-1)) + x_skip
        y1 = self.proj_y1(torch.cat([y_skip, pro1.permute(0, 2, 1).expand_as(y_skip)], dim=-1))
        y2 = self.proj_y2(torch.cat([y_skip, pro2.permute(0, 2, 1).expand_as(y_skip)], dim=-1))
        y_merge = self.final_proj_y(torch.cat([y1, y2], dim=-1)) + y_skip

        # ffn
        x = x + self.ffn_x(self.norm2(x_merge))
        y = y + self.ffn_y(self.norm2(y_merge))
        return x, y, pseudo_mask


class MixVisionTransformer(BaseModule):
    def __init__(self,
                 shot=1,
                 in_channels=64,
                 num_similarity_channels = 2,
                 num_down_stages = 3,
                 embed_dims = 64,
                 num_heads = [2, 4, 8],
                 match_dims = 64,
                 match_nums_heads = 2,
                 down_patch_sizes = [1, 3, 3],
                 down_stridess = [1, 2, 2],
                 down_sr_ratio = [4, 2, 1],
                 mlp_ratio=4,
                 drop_rate=0.1,
                 attn_drop_rate=0.,
                 qkv_bias=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        super(MixVisionTransformer, self).__init__(init_cfg=init_cfg)
        self.shot = shot

        self.num_similarity_channels = num_similarity_channels
        self.num_down_stages = num_down_stages
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.match_dims = match_dims
        self.match_nums_heads = match_nums_heads
        self.down_patch_sizes = down_patch_sizes
        self.down_stridess = down_stridess
        self.down_sr_ratio = down_sr_ratio
        self.mlp_ratio=mlp_ratio
        self.qkv_bias = qkv_bias

        # ========================================
        # Self attention
        # ========================================
        self.down_sample_layers = ModuleList()
        for i in range(num_down_stages):
            self.down_sample_layers.append(nn.ModuleList([
                PatchEmbed(
                    in_channels=embed_dims,
                    embed_dims=embed_dims,
                    kernel_size=down_patch_sizes[i],
                    stride=down_stridess[i],
                    padding=down_stridess[i] // 2,
                    norm_cfg=norm_cfg),
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=down_sr_ratio[i]),
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=down_sr_ratio[i]),
                build_norm_layer(norm_cfg, embed_dims)[1]
            ]))

        # ========================================
        # Cross attention
        # ========================================
        self.match_layers = ModuleList()
        for i in range(self.num_down_stages):
            level_match_layers = ModuleList([
                PrototypeEnhanced(self.match_dims, drop=0., mlp_ratio=mlp_ratio),
                TransformerEncoderLayer(
                    embed_dims=self.match_dims,
                    num_heads=self.match_nums_heads,
                    feedforward_channels=self.mlp_ratio * self.match_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=1
                ),
                ConvModule(self.match_dims + self.num_similarity_channels, self.match_dims, kernel_size=3, stride=1, padding=1, norm_cfg=dict(type="SyncBN"))])
            self.match_layers.append(level_match_layers)

        # ========================================
        # Output
        # ========================================
        self.parse_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims),
            nn.ReLU()
        ) for _ in range(self.num_down_stages)
        ])
        self.cls = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, 2, kernel_size=1, stride=1, padding=0)
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(MixVisionTransformer, self).init_weights()

    def forward(self, q_x, s_x, mask, similarity, pro_pred):
        # q_x: (bs, c, h, w)
        # s_x: (bs*shot, c, h, w)
        # mask: (bs*shot, 1, h, w)
        # similarity: (bs, 1, h, w)
        # pro_pred: (bs, 1, h, w)

        # ========================================
        # Self attention
        # ========================================
        down_query_features = []
        down_support_features = []
        hw_shapes = []
        down_masks = []
        ae_masks = []  # for ambiguity eliminator
        down_similarity = []
        down_pro_pred = []
        ae_pro_pred_list = []
        weights = []
        for i, layer in enumerate(self.down_sample_layers):
            # Patch embed
            q_x, q_hw_shape = layer[0](q_x)
            s_x, s_hw_shape = layer[0](s_x)

            # Self attentions
            tmp_mask = resize(mask, s_hw_shape, mode="nearest")
            tmp_pro_pred = resize(pro_pred.unsqueeze(1), s_hw_shape, mode="nearest")
            ae_mask = rearrange(tmp_mask, '(b n) 1 h w -> b (n h w) 1', n=self.shot)
            ae_masks.append(ae_mask)  # for ambiguity eliminator
            q_x, s_x = layer[1](q_x, hw_shape=q_hw_shape)[0], layer[1](s_x, hw_shape=s_hw_shape, mask=ae_mask)[0]
            q_x, s_x = layer[2](q_x, hw_shape=q_hw_shape)[0], layer[2](s_x, hw_shape=s_hw_shape, mask=ae_mask)[0]
            q_x, s_x = layer[3](q_x), layer[3](s_x)

            tmp_mask = rearrange(tmp_mask, "(b n) 1 h w -> b 1 (n h w)", n=self.shot)
            tmp_pro_pred = rearrange(tmp_pro_pred, "b 1 h w -> b 1 (h w)")

            tmp_similarity = resize(similarity, q_hw_shape, mode="bilinear", align_corners=True)
            ae_pro_pred = resize(pro_pred.unsqueeze(1), q_hw_shape, mode="bilinear", align_corners=True)
            down_query_features.append(q_x)
            down_support_features.append(rearrange(s_x, "(b n) l c -> b (n l) c", n=self.shot))
            hw_shapes.append(q_hw_shape)
            down_masks.append(tmp_mask)
            down_pro_pred.append(tmp_pro_pred)
            down_similarity.append(tmp_similarity)
            ae_pro_pred_list.append(ae_pro_pred)
            if i != self.num_down_stages - 1:
                q_x, s_x = nlc_to_nchw(q_x, q_hw_shape), nlc_to_nchw(s_x, s_hw_shape)

        # ========================================
        # Cross attention
        # ========================================
        outs = None
        pseudo_masks = []
        for i in range(self.num_down_stages).__reversed__():
            layer = self.match_layers[i]  # 0 - ambiguity eliminator; 1 - cross attention ...

            # ========================================
            # Ambiguity eliminator
            # x: (n, h*w, c)
            # ========================================
            down_query_features[i], down_support_features[i], pseudo_mask = layer[0](
                down_query_features[i], down_support_features[i],
                hw_shapes[i], ae_masks[i], ae_pro_pred_list[i]
            )

            pseudo_masks.append(pseudo_mask)

            # ========================================
            # Cross attention
            # ========================================
            out, weight = layer[1](
                x=down_query_features[i],
                hw_shape=hw_shapes[i],
                source=down_support_features[i],
                mask=down_masks[i],
                cross=True,
                pro_pred=down_pro_pred[i]
            )
            out = nlc_to_nchw(out, hw_shapes[i])
            weight = weight.view(out.shape[0], hw_shapes[i][0], hw_shapes[i][1])
            out = layer[2](torch.cat([out, down_similarity[i]], dim=1))
            weights.append(weight)
            if outs is None:
                outs = self.parse_layers[i](out)
            else:
                outs = resize(outs, size=out.shape[-2:], mode="bilinear")
                outs = outs + self.parse_layers[i](out + outs)
        outs = self.cls(outs)
        return outs, weights, pseudo_masks


class Transformer(nn.Module):
    def __init__(self, shot=1) -> None:
        super().__init__()
        self.shot=shot
        self.mix_transformer = MixVisionTransformer(shot=self.shot)

    def forward(self, features, supp_features, mask, similaryty, pro_pred):
        # features: (bs, c, h, w)
        # supp_features: (bs*shot, c, h, w)
        # mask: (bs*shot, 1, h, w)
        # similarity: (bs, 1, h, w)
        shape = features.shape[-2:]
        outs, weights, pseudo_masks = self.mix_transformer(features, supp_features, mask, similaryty, pro_pred)
        return outs, weights, pseudo_masks
