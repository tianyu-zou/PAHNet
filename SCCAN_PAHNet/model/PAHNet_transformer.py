import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CrossWindowAttention(nn.Module):
    """ Window based multi-head cross attention (CW-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., alpha=2):
        super(CrossWindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_q = nn.Linear(1, dim, bias=qkv_bias)
        self.proj_s = nn.Linear(1, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = alpha
        self.qkv_bias = qkv_bias

    def forward(self, q, s, q_mask=None, s_mask=None, pro_pred=None):
        """ Forward function.
        Args:
            q: input query features with shape of (num_windows*B, N, C)
            s: input query/support features with shape of (num_windows*B, 2*N, C)
            s_mask: input cross masks with shape of (num_windows*B, N, 1)
            q_mask: input self masks with shape of (num_windows*B, N, 1)
            pro_pred: input prototype method query pred with shape of (num_windows*B, N, 1)
        """
        B_, N, C = q.shape
        N_kv = s.size(1)

        q = self.q(q).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                     4)  # 1, B_, #HEADS, N, C // #HEADS

        kv = self.kv(s)
        kv = kv.reshape(B_, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                          4)  # 2, B_, #HEADS, 2N, C // #HEADS

        q = q[0]
        k, v = kv[0], kv[1]

        # Use attention for self-attention, use cosine similarity for cross-attention
        attn = (q @ k.transpose(-2, -1))  # B_, #HEADS, N, 2N
        attn_self = attn[:, :, :, :N]  # B_, #HEADS, N, N
        attn_self = attn_self * self.scale

        # Only activate FG-FG and BG-BG interactions
        if q_mask is not None:
            q_mask = q_mask.unsqueeze(1).clone()  # B_, 1, N, 1
            q_mask[q_mask >= 0.5] = 1
            q_mask[q_mask < 0.5] = 0

            hor_mask = q_mask.clone()
            ver_mask = q_mask.clone().transpose(-2, -1)  # B_, 1, 1, N

            mask = hor_mask + ver_mask
            mask[mask == 2] = 0

            mask[mask == 1] = float('-inf')

            attn_self += mask

        attn_cross = attn[:, :, :, N:]  # B_, #HEADS, N, N
        cos_eps = 1e-7
        q_norm = torch.norm(q, 2, 3, True)
        k_norm = torch.norm(k[:, :, N:, :], 2, 3, True)
        attn_cross = attn_cross / (q_norm @ k_norm.transpose(-2, -1) + cos_eps)

        # Modified query cross attention map
        if s_mask is not None:
            s_mask = s_mask.unsqueeze(1).clone()  # B_, 1, N, 1

            # s_mask is interpolated and no longer binary. A threshold of 0.5 is introduced for binarization.
            s_mask[s_mask >= 0.5] = 1
            s_mask[s_mask < 0.5] = 0
            q_mask = pro_pred.unsqueeze(1).clone()  # B_, 1, N, 1
            q_mask[q_mask >= 1] = 1

            hor_mask = q_mask.clone()
            ver_mask = s_mask.clone().transpose(-2, -1)

            hor_mask[q_mask < 0.3] = 0
            hor_mask[q_mask > 0.7] = 1

            # mask matrix
            hy_mask = hor_mask + ver_mask
            hy_mask[hy_mask == 1] = float('-inf')
            hy_mask[hy_mask > 0] = 0

            s_mask = self.proj_s(s_mask).reshape(B_, 1, N, self.num_heads, C // self.num_heads).permute(1, 0, 3, 2, 4)[0]  # B_, 1, N, H, c
            q_mask = self.proj_q(q_mask).reshape(B_, 1, N, self.num_heads, C // self.num_heads).permute(1, 0, 3, 2, 4)[0]  # B_, 1, N, H, c

            # re-weight matrix
            mask = q_mask @ s_mask.transpose(-2, -1)   # B_, 1, N, N

            attn_cross = mask * attn_cross + hy_mask

        attn = torch.cat([attn_self, attn_cross], dim=-1)  # B_, #HEADS, N, 2N

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        q = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        q = self.proj(q)
        q = self.proj_drop(q)
        return q


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


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # PrototypeEnhanced Adaptor
        self.pe = PrototypeEnhanced(dim, drop=drop)

        # SCCAN
        self.norm1 = norm_layer(dim)

        self.attn_q = CrossWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_s = CrossWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_q = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_s = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]  # bs, 1, -1
        expanse = list(input.size())  # bs, c, hw
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)  # bs, c, hw
        return torch.gather(input, dim, index)

    def generate_indices(self, q, s, s_mask=None, mask_bg=True):
        bs, c, _, _ = q.size()
        window_size = self.window_size

        q_protos = F.avg_pool2d(
            q,
            kernel_size=(window_size, window_size),
            stride=(window_size, window_size)
        )  # bs, c, n_hn_w
        s_mask_protos = None
        gap_eps = 5e-4
        if s_mask is not None and mask_bg:
            s_mask_protos = F.avg_pool2d(
                s_mask,
                kernel_size=(window_size, window_size),
                stride=(window_size, window_size)
            ) * window_size * window_size + gap_eps
            s_protos = F.avg_pool2d(
                s * s_mask,
                kernel_size=(window_size, window_size),
                stride=(window_size, window_size)
            ) * window_size * window_size / s_mask_protos  # bs, c, n_hn_w
        else:
            s_protos = F.avg_pool2d(
                s,
                kernel_size=(self.window_size, self.window_size),
                stride=(self.window_size, self.window_size)
            )  # bs, c, n_hn_w

        q_protos = q_protos.view(bs, c, -1)  # bs, c, hw
        s_protos = s_protos.view(bs, c, -1).permute(0, 2, 1)  # bs, hw, c
        if s_mask is not None and mask_bg:
            s_mask_protos = s_mask_protos.view(bs, 1, -1).permute(0, 2, 1)  # bs, hw, 1
            s_mask_protos[s_mask_protos != gap_eps] = 1
            s_mask_protos[s_mask_protos == gap_eps] = 0

        q_protos_norm = torch.norm(q_protos, 2, 1, True)
        s_protos_norm = torch.norm(s_protos, 2, 2, True)

        cos_eps = 1e-7
        cos_sim = torch.bmm(s_protos, q_protos) / (torch.bmm(s_protos_norm, q_protos_norm) + cos_eps)
        if s_mask_protos is not None:
            cos_sim = (cos_sim + 1) / 2
            cos_sim = cos_sim * s_mask_protos
        cos_sim_star, cos_sim_star_index = torch.max(cos_sim, dim=1)
        return cos_sim_star_index

    def forward(self, q, s, s_mask, pro_pred):
        """ Forward function.
        Args:
            q: Input query feature, tensor size (B, H*W, C).
            s: Input support feature, tensor size (B, H*W, C).
            s_mask: Input support feature, tensor size (B, H*W, 1).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = q.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        # ========================================
        # PrototypeEnhanced Adaptor
        # ========================================
        q, s, pseudo_mask = self.pe(q, s, (H, W), s_mask, pro_pred.unsqueeze(1))

        shortcut_q = q
        shortcut_s = s
        q = self.norm1(q)
        s = self.norm1(s)
        q = q.view(B, H, W, C)
        s = s.view(B, H, W, C)
        s_mask = s_mask.view(B, H, W, 1)
        pro_pred = pro_pred.view(B, H, W, 1).float()
        _, Hp, Wp, _ = q.shape

        # ========================================
        # Self/Cross-attention
        # ========================================
        if self.shift_size > 0:
            pad_l = pad_t = self.window_size // 2
            pad_r = pad_b = self.window_size - (self.window_size // 2)
            shifted_q = F.pad(q, (0, 0, pad_l, pad_r, pad_t, pad_b))
            shifted_s = F.pad(s, (0, 0, pad_l, pad_r, pad_t, pad_b))
            shifted_s_mask = F.pad(s_mask, (0, 0, pad_l, pad_r, pad_t, pad_b))
            pro_pred = F.pad(pro_pred, (0, 0, pad_l, pad_r, pad_t, pad_b))
            Hp += self.window_size
            Wp += self.window_size
        else:
            shifted_q = q
            shifted_s = s
            shifted_s_mask = s_mask
            pro_pred = pro_pred

        # ====================
        # Self+Cross-attention for query
        # ====================
        qs_index = self.generate_indices(shifted_q.permute(0, 3, 1, 2), shifted_s.permute(0, 3, 1, 2),
                                         s_mask=shifted_s_mask.permute(0, 3, 1, 2),
                                         mask_bg=True)

        s_clone = shifted_s.permute(0, 3, 1, 2)
        s_mask_clone = shifted_s_mask.permute(0, 3, 1, 2)

        s_unfold = F.unfold(
            s_clone,
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size)
        )
        s_mask_unfold = F.unfold(
            s_mask_clone,
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size)
        )

        s_unfold_tsf = self.bis(s_unfold, 2, qs_index)
        s_mask_unfold_tsf = self.bis(s_mask_unfold, 2, qs_index)

        s_fold = F.fold(
            s_unfold_tsf,
            output_size=(Hp, Wp),
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size)
        )
        s_mask_fold = F.fold(
            s_mask_unfold_tsf,
            output_size=(Hp, Wp),
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size)
        )

        shifted_s_tsf = s_fold.permute(0, 2, 3, 1)
        shifted_s_mask_tsf = s_mask_fold.permute(0, 2, 3, 1)

        q_windows = window_partition(shifted_q, self.window_size)
        q_windows = q_windows.view(-1, self.window_size * self.window_size, C)

        pro_pred = window_partition(pro_pred, self.window_size)
        pro_pred = pro_pred.view(-1, self.window_size * self.window_size, 1)

        s_windows = window_partition(shifted_s_tsf, self.window_size)
        s_windows = s_windows.view(-1, self.window_size * self.window_size, C)

        s_mask_windows = window_partition(shifted_s_mask_tsf, self.window_size)
        s_mask_windows = s_mask_windows.view(-1, self.window_size * self.window_size, 1)

        qs_windows = torch.cat([q_windows, s_windows], dim=1)
        attn_q_windows = self.attn_q(q_windows, qs_windows, s_mask=s_mask_windows, pro_pred=pro_pred)

        attn_q_windows = attn_q_windows.view(-1, self.window_size, self.window_size, C)
        shifted_q = window_reverse(attn_q_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            q = shifted_q[:, pad_l: pad_l + H, pad_t: pad_t + W, :].contiguous()  # bs, h, w, c
        else:
            q = shifted_q
        q = q.view(B, H * W, C)
        q = shortcut_q + self.drop_path(q)
        q = q + self.drop_path(self.mlp_q(self.norm2(q)))

        # ====================
        # Self-attention for support
        # ====================
        s_windows = window_partition(shifted_s, self.window_size)
        s_windows = s_windows.view(-1, self.window_size * self.window_size, C)

        s_mask_windows = window_partition(shifted_s_mask, self.window_size)
        s_mask_windows = s_mask_windows.view(-1, self.window_size * self.window_size, 1)
        attn_s_windows = self.attn_s(s_windows, s_windows, q_mask=s_mask_windows)
        attn_s_windows = attn_s_windows.view(-1, self.window_size, self.window_size, C)
        shifted_s = window_reverse(attn_s_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            s = shifted_s[:, pad_l: pad_l + H, pad_t: pad_t + W, :].contiguous()  # bs, h, w, c
        else:
            s = shifted_s
        s = s.view(B, H * W, C)
        s = shortcut_s + self.drop_path(s)
        s = s + self.drop_path(self.mlp_s(self.norm2(s)))

        return q, s, pseudo_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm
                 ):
        super(BasicLayer, self).__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)])

    def forward(self, q, s, s_mask, H, W, pro_pred):
        """ Forward function.
        Args:
            q: Input query feature, tensor size (B, H*W, C).
            s: Input support feature, tensor size (B, H*W, C).
            s_mask: Input support mask, tensor size (B, H*W, 1).
            H, W: Spatial resolution of the input feature.
        """
        # calculate attention mask for SW-MSA
        pseudo_masks = []
        for blk in self.blocks:
            blk.H, blk.W = H, W
            q, s, pseudo_mask = blk(q, s, s_mask, pro_pred)
            pseudo_masks.append(pseudo_mask)
        return q, s, pseudo_masks


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=1.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1
                 ):
        super(SwinTransformer, self).__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be None')

    def forward(self, q, s, s_mask, pro_pred):
        """Forward function."""
        Wh, Ww = q.size(2), q.size(3)
        q = q.flatten(2).transpose(1, 2)  # bs, hw, c
        s = s.flatten(2).transpose(1, 2)  # bs, hw, c
        s_mask = s_mask.flatten(2).transpose(1, 2)  # bs, hw, 1
        q = self.pos_drop(q)
        s = self.pos_drop(s)

        outs = []
        pseudo_masks = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            q, s, pseudo_mask = layer(q, s, s_mask, Wh, Ww, pro_pred)
            pseudo_masks.extend(pseudo_mask)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                q_out = norm_layer(q)
                out = q_out.view(-1, Wh, Ww, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs), tuple(pseudo_masks)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
