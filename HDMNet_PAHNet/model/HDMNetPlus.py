import torch
from torch import nn
import torch.nn.functional as F
from HDMNet_PAHNet.model.TransformerPlus import Transformer
import HDMNet_PAHNet.model.resnet as models
from HDMNet_PAHNet.model.PSPNet import OneModel as PSPNet
from einops import rearrange
from HDMNet_PAHNet.model.SSP_matching import SSP_MatchingNet as SSP_matching


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_similarity(q, s, mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate((mask == 1).float(), q.shape[-2:])
    cosine_eps = 1e-7
    s = s * mask
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity


def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # C*N
    fea_T = fea.permute(0, 2, 1)  # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()
        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.dataset = args.data_set
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        self.low_fea_id = args.low_fea[-1]

        assert args.layers in [50, 101, 152]
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.shot = args.shot
        self.vgg = args.vgg
        models.BatchNorm = BatchNorm

        # ========================================
        # Pretrained backbone
        # ========================================
        PSPNet_ = PSPNet(args)
        new_param = torch.load(args.pre_weight, map_location=torch.device('cpu'))['state_dict']
        print(args.pre_weight)
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4
        self.ppm = PSPNet_.ppm
        self.cls = nn.Sequential(PSPNet_.cls[0], PSPNet_.cls[1])
        self.base_learnear = nn.Sequential(PSPNet_.cls[2], PSPNet_.cls[3], PSPNet_.cls[4])

        # ========================================
        # Pretrained prototype predictor
        # ========================================
        self.ssp = SSP_matching('resnet50', False)
        ssp_weight = "../initmodel/SSPNet/{}/split{}/{}/best.pth".format(self.dataset, args.split,
                                                                         "resnet50" if not args.vgg else "vgg")
        ssp_param = torch.load(ssp_weight)
        try:
            self.ssp.load_state_dict(ssp_param)
        except RuntimeError:
            for key in list(ssp_param.keys()):
                ssp_param[key[7:]] = ssp_param.pop(key)
            self.ssp.load_state_dict(ssp_param)
        # ========================================
        # Feature preparation
        # ========================================
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        # ========================================
        # Prototype and prior fusion
        # ========================================
        self.query_merge = nn.Sequential(
            nn.Conv2d(512 + 2, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.supp_merge = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        # ========================================
        # Transformer
        # ========================================
        self.transformer = Transformer(shot=self.shot)

        # ========================================
        # Base class ensemble
        # ========================================
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))
        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))

        # ========================================
        # K-Shot reweighting
        # ========================================
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

    def cos_sim(self, query_feat_high, tmp_supp_feat, cosine_eps=1e-7):
        q = query_feat_high.flatten(2).transpose(-2, -1)
        s = tmp_supp_feat.flatten(2).transpose(-2, -1)

        tmp_query = q
        tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s
        tmp_supp = tmp_supp.contiguous()
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        return similarity

    def generate_prior_proto(self, query_feat_high, final_supp_list, mask_list, fts_size):
        bsize, ch_sz, sp_sz, _ = query_feat_high.size()[:]
        fg_list = []
        bg_list = []
        fg_sim_maxs = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            fg_supp_feat = Weighted_GAP(tmp_supp_feat, tmp_mask)
            bg_supp_feat = Weighted_GAP(tmp_supp_feat, 1 - tmp_mask)

            fg_sim = self.cos_sim(query_feat_high, fg_supp_feat, cosine_eps)
            bg_sim = self.cos_sim(query_feat_high, bg_supp_feat, cosine_eps)

            fg_sim = fg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
            bg_sim = bg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)

            fg_sim_max = fg_sim.max(1)[0]  # bsize
            fg_sim_maxs.append(fg_sim_max.unsqueeze(-1))  # bsize, 1

            fg_sim = (fg_sim - fg_sim.min(1)[0].unsqueeze(1)) / (
                    fg_sim.max(1)[0].unsqueeze(1) - fg_sim.min(1)[0].unsqueeze(1) + cosine_eps)

            bg_sim = (bg_sim - bg_sim.min(1)[0].unsqueeze(1)) / (
                    bg_sim.max(1)[0].unsqueeze(1) - bg_sim.min(1)[0].unsqueeze(1) + cosine_eps)

            fg_sim = fg_sim.view(bsize, 1, sp_sz, sp_sz)
            bg_sim = bg_sim.view(bsize, 1, sp_sz, sp_sz)

            fg_sim = F.interpolate(fg_sim, size=fts_size, mode='bilinear', align_corners=True)
            bg_sim = F.interpolate(bg_sim, size=fts_size, mode='bilinear', align_corners=True)
            fg_list.append(fg_sim)
            bg_list.append(bg_sim)
        fg_corr = torch.cat(fg_list, 1)  # bsize, shots, h, w
        bg_corr = torch.cat(bg_list, 1)
        corr = (fg_corr - bg_corr)
        corr[corr < 0] = 0
        corr_max = corr.view(bsize, len(final_supp_list), -1).max(-1)[0]  # bsize, shots

        fg_sim_maxs = torch.cat(fg_sim_maxs, dim=-1)  # bsize, shots
        return fg_corr, bg_corr, corr, fg_sim_maxs, corr_max

    def forward(self, x, y_m=None, y_b=None, s_x=None, s_y=None, cat_idx=None):
        self.ssp.eval()
        img_s = []
        mask_s = []
        for k in range(self.shot):
            img_s.append(s_x[:, k, :, :, :])
            mask_s.append(s_y[:, k, :, :])
        with torch.no_grad():
            pro_pred = self.ssp(img_s, mask_s, x, None)[0]
            pro_pred = torch.softmax(pro_pred, 1)
        bs = x.shape[0]
        h, w = x.shape[-2:]
        # ========================================
        # Feature extraction
        # ========================================
        s_x = rearrange(s_x, "b n c h w -> (b n) c h w")
        with torch.no_grad():
            _, _, query_feat_2, query_feat_3, query_feat_4, query_feat_5 = self.extract_feats(x)
            _, _, supp_feat_2, supp_feat_3, supp_feat_4, supp_feat_5 = self.extract_feats(s_x)

        # ========================================
        # Feature preparation
        # ========================================
        if self.vgg:
            query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                         mode='bilinear', align_corners=True)
            supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                        align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)
        supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
        supp_feat = self.down_supp(supp_feat)

        # ========================================
        # Support prototype extraction
        # ========================================
        mask = rearrange(s_y, "b n h w -> (b n) 1 h w")
        mask = (mask == 1).float()
        supp_mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                  align_corners=True)  # (bs*shot, 1, h, w)
        supp_feat_bin = Weighted_GAP(supp_feat, supp_mask)  # (bs*shot, c, 1, 1)
        supp_feat_bin = supp_feat_bin.repeat(1, 1, supp_feat.shape[-2], supp_feat.shape[-1])  # (bs*shot, c, h, w)

        # ========================================
        # Low-level support features
        # For base class ensemble
        # ========================================
        supp_feat_item = eval('supp_feat_' + self.low_fea_id)
        supp_feat_item = rearrange(supp_feat_item, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_list = [supp_feat_item[:, i, ...] for i in range(self.shot)]

        # ========================================
        # Prior masks
        # ========================================
        fts_size = query_feat.size()[-2:]
        if self.shot == 1:
            corr_fg_4, _, corr_4, corr_fg_4_sim_max, corr_4_sim_max = self.generate_prior_proto(query_feat_4,
                                                                                                [supp_feat_4],
                                                                                                [supp_mask], fts_size)
            corr_fg_5, _, corr_5, corr_fg_5_sim_max, corr_5_sim_max = self.generate_prior_proto(query_feat_5,
                                                                                                [supp_feat_5],
                                                                                                [supp_mask], fts_size)
        else:
            supp_mask = rearrange(supp_mask, "(b n) c h w -> b n c h w", n=self.shot)
            supp_feat_4 = rearrange(supp_feat_4, "(b n) c h w -> b n c h w", n=self.shot)
            supp_feat_5 = rearrange(supp_feat_5, "(b n) c h w -> b n c h w", n=self.shot)

            corr_fg_4, _, corr_4, corr_fg_4_sim_max, corr_4_sim_max = self.generate_prior_proto(
                query_feat_4,
                [supp_feat_4[:, i, :, :, :] for i in range(self.shot)],
                [supp_mask[:, i, :, :, :] for i in range(self.shot)],
                fts_size
            )
            corr_fg_5, _, corr_5, corr_fg_5_sim_max, corr_5_sim_max = self.generate_prior_proto(
                query_feat_5,
                [supp_feat_5[:, i, :, :, :] for i in range(self.shot)],
                [supp_mask[:, i, :, :, :] for i in range(self.shot)],
                fts_size
            )

        corr_fg = corr_fg_4.clone()  # bs, shots, h, w
        corr = corr_4.clone()  # bs, shots, h, w
        for i in range(bs):
            for j in range(self.shot):
                if corr_fg_4_sim_max[i, j] < corr_fg_5_sim_max[i, j]:
                    corr_fg[i, j] = corr_fg_5[i, j]
                if corr_4_sim_max[i, j] < corr_5_sim_max[i, j]:
                    corr[i, j] = corr_5[i, j]
        corr_fg = corr_fg.mean(1, True)
        corr = corr.mean(1, True)
        similarity = torch.cat([corr_fg, corr], dim=1)  # (bs, 1, h, w)

        # ========================================
        # Prototype and prior fusion
        # ========================================
        supp_feat = self.supp_merge(torch.cat([supp_feat, supp_feat_bin], dim=1))
        supp_feat_bin = rearrange(supp_feat_bin, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_bin = torch.mean(supp_feat_bin, dim=1)
        query_feat = self.query_merge(torch.cat([query_feat, supp_feat_bin, similarity * 10], dim=1))

        # ========================================
        # Transformer
        # ========================================
        # mask 是support的mask
        # similarity是query的prior mask
        # pro_pred是query 的prototype的预测
        meta_out, weights, pseudo_masks = self.transformer(query_feat, supp_feat, mask, similarity,
                                                           pro_pred[:, 1, :, :])

        # ========================================
        # Base output
        # ========================================
        base_out = self.base_learnear(query_feat_5)

        # ========================================
        # Output ensemble
        # ========================================
        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # K-Shot Reweighting
        bs = x.shape[0]
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id))  # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            idx3 = idx1.gather(1, idx2)
            weight = weight.gather(1, idx3)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

        # Following the implementation of BAM ( https://github.com/chunbolang/BAM )
        meta_map_bg = meta_out_soft[:, 0:1, :, :]
        meta_map_fg = meta_out_soft[:, 1:, :, :]
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes + 1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array != 0) & (c_id_array != c_id)
                base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
            base_map = torch.cat(base_map_list, 0)
        else:
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Interpolate
        meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
        base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
        final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        # ========================================
        # Loss
        # ========================================
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            aux_loss1 = self.criterion(meta_out, y_m.long())
            aux_loss2 = self.criterion(base_out, y_b.long())
            aux_loss3 = torch.zeros_like(main_loss)
            # ========================================
            # Loss
            # ========================================
            for pseudo_mask in pseudo_masks:
                gt_mask = F.interpolate(y_m.unsqueeze(1).float(), size=pseudo_mask.size()[-2:], mode='nearest').squeeze(
                    1)
                aux_loss3 += self.criterion(pseudo_mask, gt_mask.long())
            aux_loss3 /= len(pseudo_masks)

            weight_t = (y_m == 1).float()
            weight_t = torch.masked_fill(weight_t, weight_t == 0, -1e9)
            for i, weight in enumerate(weights):
                if i == 0:
                    distil_loss = self.disstil_loss(weight_t, weight)
                else:
                    distil_loss += self.disstil_loss(weight_t, weight)
                weight_t = weight.detach()

            return final_out.max(1)[1], main_loss + aux_loss1, distil_loss / 3, aux_loss2 + aux_loss3
        else:
            return final_out, meta_out, base_out, pro_pred

    def disstil_loss(self, t, s):
        if t.shape[-2:] != s.shape[-2:]:
            t = F.interpolate(t.unsqueeze(1), size=s.shape[-2:], mode='bilinear').squeeze(1)
        t = rearrange(t, "b h w -> b (h w)")
        s = rearrange(s, "b h w -> b (h w)")
        s = torch.softmax(s, dim=1)
        t = torch.softmax(t, dim=1)
        loss = t * torch.log(t + 1e-12) - t * torch.log(s + 1e-12)
        loss = loss.sum(1).mean()
        return loss

    def get_optim(self, model, args, LR):
        params = [
            {'params': model.transformer.mix_transformer.parameters()},
            {'params': model.down_supp.parameters(), "lr": LR * 10},
            {'params': model.down_query.parameters(), "lr": LR * 10},
            {'params': model.supp_merge.parameters(), "lr": LR * 10},
            {'params': model.query_merge.parameters(), "lr": LR * 10},
            {'params': model.gram_merge.parameters(), "lr": LR * 10},
            {'params': model.cls_merge.parameters(), "lr": LR * 10},
        ]
        if self.shot > 1:
            params.append({'params': model.kshot_rw.parameters(), 'lr': LR * 10})
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        return optimizer

    def freeze_modules(self, model):
        for param in model.ssp.parameters():
            param.requires_grad = False
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.cls.parameters():
            param.requires_grad = False
        for param in model.base_learnear.parameters():
            param.requires_grad = False

    def extract_feats(self, x, mask=None):
        results = []
        with torch.no_grad():
            if mask is not None:
                tmp_mask = F.interpolate(mask, size=x.shape[-2], mode='nearest')
                x = x * tmp_mask
            feat = self.layer0(x)
            results.append(feat)
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for _, layer in enumerate(layers):
                feat = layer(feat)
                results.append(feat.clone())
            feat = self.ppm(feat)
            feat = self.cls(feat)
            results.append(feat)
        return results
