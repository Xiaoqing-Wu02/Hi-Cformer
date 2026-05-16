import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import block_encoder, BlockWithCustomMask


class hicformer(nn.Module):
    def __init__(
        self,
        whole_size=1000,
        img_size=112,
        hidden_dim=60,
        patch_size=[8, 16, 32, 64, 128],
        encoder_dim=[],
        chr_num=21,
        in_chans=1,
        embed_dim=128,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        chr_mutual_visibility=True,
        weight=0.1,
        chr_single_loss=False,
        otherlf="MSE",
        enorm=False,
        num_patches=24,
        weight_mode="mask",
    ):
        super().__init__()

        self.chr_single_loss = chr_single_loss
        self.embed_dim = embed_dim
        self.chr_mutual_visibility = chr_mutual_visibility
        self.patch_size = patch_size
        self.whole_size = whole_size
        self.chr_num = chr_num
        self.img_size = img_size
        self.weight = weight
        self.num_patches = num_patches
        self.otherlf = otherlf
        self.enorm = enorm
        self.weight_mode = weight_mode

        self.multi_encoder = nn.ModuleList(
            [block_encoder(s, in_chans, embed_dim, stride=s, padding=0, norm_layer=norm_layer) for s in patch_size]
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Embedding(num_embeddings=num_patches + 1, embedding_dim=embed_dim)
        self.size_embed = nn.Embedding(num_embeddings=len(self.multi_encoder), embedding_dim=embed_dim)
        self.chr_embed = nn.Embedding(num_embeddings=chr_num, embedding_dim=embed_dim)

        self.transformer_blocks = nn.ModuleList(
            [BlockWithCustomMask(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)]
        )
        self.norm = norm_layer(embed_dim)

        prev_dim = chr_num * embed_dim
        latent_layer = []
        for dim in encoder_dim:
            latent_layer.append(nn.Linear(prev_dim, dim, bias=True))
            latent_layer.append(nn.GELU())
            prev_dim = dim

        self.pred_hidden = nn.Linear(prev_dim, hidden_dim, bias=True)
        self.latent_layer = nn.Sequential(*latent_layer)

        self.cell_decoder = nn.Sequential(nn.Linear(hidden_dim, whole_size, bias=True), nn.ReLU())
        self.block_decoder = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, s**2, bias=True), nn.ReLU()) for s in patch_size])
        self.chr_decoder = nn.Sequential(nn.Linear(embed_dim, int(img_size * (img_size + 1) / 2), bias=True), nn.ReLU())

        # ---------- buffers for fast expand_tensor ----------
        tri = torch.ones(img_size, img_size, dtype=torch.bool).triu()
        self.register_buffer("_tri_mask_flat", tri.flatten(), persistent=False)

        # ---------- buffers for fixed token layout (built lazily) ----------
        self.register_buffer("_layout_ready", torch.tensor(False), persistent=False)
        self.register_buffer("_layout_signature", torch.empty(0, dtype=torch.int32), persistent=False)  # store lens signature

        self.register_buffer("_pos_index", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_size_index", torch.empty(0, dtype=torch.long), persistent=False)  # -1 means "no size embed"
        self.register_buffer("_chr_index", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_size_mask_f", torch.empty(0, dtype=torch.float32), persistent=False)  # 0/1 float mask

        self.register_buffer("_attn_mask", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("_token_class", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("_chr_cls_index", torch.empty(0, dtype=torch.long), persistent=False)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.pos_embed.weight)
        torch.nn.init.xavier_uniform_(self.size_embed.weight)
        torch.nn.init.xavier_uniform_(self.chr_embed.weight)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        for i in range(len(self.multi_encoder)):
            w = self.multi_encoder[i].proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # -----------------------------
    # Fast upper-triangular flatten
    # -----------------------------
    def expand_tensor(self, img):
        # img: [B, C, H, H]
        B, C, H, W = img.shape
        flat = img.view(B, C, -1)
        if H == self.img_size and W == self.img_size:
            mask_flat = self._tri_mask_flat
        else:
            mask_flat = torch.ones(H, W, device=img.device, dtype=torch.bool).triu().flatten()
        return flat[:, :, mask_flat]

    # ---------------------------------------
    # Layout cache builder (one-time per layout)
    # ---------------------------------------
    def _build_layout_cache(self, lens_per_chr_per_scale, device):
        """
        lens_per_chr_per_scale: list length chr_num_actual (len(x)-2),
            each is list length num_scales_valid_for_that_chr, containing token counts per scale.
        Assumes the concatenation order is:
            [chr_cls] + [scale0 tokens] + [scale1 tokens] + ...
        """
        # signature to detect if layout changes
        # flatten all lens into one 1D int tensor with separators encoded by 0
        sig_list = []
        for chr_lens in lens_per_chr_per_scale:
            sig_list.append(-1)  # chromosome separator
            sig_list.extend(chr_lens)
        signature = torch.tensor(sig_list, dtype=torch.int32, device=device)

        # if already built and same signature -> no-op
        if bool(self._layout_ready.item()) and self._layout_signature.numel() == signature.numel():
            if torch.equal(self._layout_signature, signature):
                return

        # Build indices
        pos_idx = []
        size_idx = []
        chr_idx = []
        token_class = []
        chr_cls_positions = []

        flag = 0
        for chr_i, chr_lens in enumerate(lens_per_chr_per_scale):
            # chr cls token
            chr_cls_positions.append(flag)
            pos_idx.append(torch.tensor([0], device=device, dtype=torch.long))
            size_idx.append(torch.tensor([-1], device=device, dtype=torch.long))  # no size embed for chr_cls
            chr_idx.append(torch.tensor([chr_i], device=device, dtype=torch.long))
            token_class.append(torch.tensor([1], device=device, dtype=torch.int8))
            flag += 1

            for s_i, l in enumerate(chr_lens):
                if l <= 0:
                    continue
                # pos indices 1..l
                pos_idx.append(torch.arange(1, 1 + l, device=device, dtype=torch.long))
                # size indices repeat s_i
                size_idx.append(torch.full((l,), s_i, device=device, dtype=torch.long))
                # chr indices repeat chr_i
                chr_idx.append(torch.full((l,), chr_i, device=device, dtype=torch.long))
                token_class.append(torch.zeros((l,), device=device, dtype=torch.int8))
                flag += l

        pos_index = torch.cat(pos_idx, dim=0)
        size_index = torch.cat(size_idx, dim=0)
        chr_index = torch.cat(chr_idx, dim=0)
        token_class = torch.cat(token_class, dim=0)
        chr_cls_index = torch.tensor(chr_cls_positions, device=device, dtype=torch.long)

        # size mask float: 1 where size_index>=0 else 0
        size_mask_f = (size_index >= 0).to(torch.float32)
        # clamp size_index for embedding lookup
        size_index_clamped = torch.clamp(size_index, min=0)

        # Build attention mask once
        L_total = pos_index.numel()
        attn_mask = torch.full((L_total, L_total), float("-inf"), dtype=torch.float32, device=device)

        # block diagonal by chromosome
        start = 0
        for chr_lens in lens_per_chr_per_scale:
            l_chr = 1 + sum(chr_lens)
            attn_mask[start : start + l_chr, start : start + l_chr] = 0.0
            start += l_chr

        if self.chr_mutual_visibility:
            idx = chr_cls_index
            attn_mask[idx[:, None], idx[None, :]] = 0.0

        # write buffers
        self._layout_signature = signature
        self._pos_index = pos_index
        self._size_index = size_index_clamped
        self._chr_index = chr_index
        self._size_mask_f = size_mask_f
        self._attn_mask = attn_mask
        self._token_class = token_class
        self._chr_cls_index = chr_cls_index
        self._layout_ready = torch.tensor(True, device=device)

    # ---------------------------------------
    # Encoder (fast path when mask_ratio==0)
    # ---------------------------------------
    def forward_encoder(self, x, mask_ratio=0.0):
        device = next(self.parameters()).device

        # --------- Fast path: layout fixed + no masking ----------
        if mask_ratio <= 0.0 and bool(self._layout_ready.item()):
            all_patches = []
            token_sequence_parts = []

            # build tokens/patches (still necessary for training loss; for inference you can ignore all_patches)
            # also we do NOT recompute any embedding indices / attn_mask here.
            for i in range(len(x) - 2):
                chr_patches = []
                chr_sentence_parts = []

                chr_cls = x[-2][:, i : i + 1, :]
                chr_sentence_parts.append(chr_cls)

                input_image = x[i]
                _, _, h, _ = input_image.shape

                # IMPORTANT: assumes same set of valid scales as cache was built with.
                for j, embed_layer in enumerate(self.multi_encoder):
                    if h >= self.patch_size[j]:
                        tokens, diagonal_patches = embed_layer(input_image)
                        chr_sentence_parts.append(tokens)
                        chr_patches.append(diagonal_patches)

                all_patches.append(chr_patches)
                token_sequence_parts.append(torch.cat(chr_sentence_parts, dim=1))

            token_sequence = torch.cat(token_sequence_parts, dim=1)  # [N, L, D]

            # vectorized embedding add
            token_sequence = token_sequence + self.pos_embed(self._pos_index).unsqueeze(0)
            token_sequence = token_sequence + self.chr_embed(self._chr_index).unsqueeze(0)
            token_sequence = token_sequence + (self.size_embed(self._size_index) * self._size_mask_f.unsqueeze(1)).unsqueeze(0)

            # transformer blocks with cached attn_mask
            attn_mask = self._attn_mask
            for blk in self.transformer_blocks:
                token_sequence = blk(token_sequence, attn_mask=attn_mask)

            return token_sequence, attn_mask, all_patches, self._token_class

        # --------- Slow path: build layout (once) + optional masking ----------
        all_patches = []
        token_sequence_parts = []
        lens_per_chr_per_scale = []

        for i in range(len(x) - 2):
            chr_patches = []
            chr_sentence_parts = []
            chr_lens = []

            chr_cls = x[-2][:, i : i + 1, :]
            chr_sentence_parts.append(chr_cls)

            input_image = x[i]
            _, _, h, _ = input_image.shape

            for j, embed_layer in enumerate(self.multi_encoder):
                if h >= self.patch_size[j]:
                    tokens, diagonal_patches = embed_layer(input_image)
                    chr_sentence_parts.append(tokens)
                    chr_patches.append(diagonal_patches)
                    chr_lens.append(tokens.shape[1])

            all_patches.append(chr_patches)
            lens_per_chr_per_scale.append(chr_lens)

            chr_sentence = torch.cat(chr_sentence_parts, dim=1)
            if mask_ratio > 0.0:
                # masking path (kept, but only used for pretrain; not your benchmark)
                N, L, D = chr_sentence.shape
                len_keep = int(L * (1 - mask_ratio))
                noise = torch.rand(N, L, device=device)
                ids_shuffle = torch.argsort(noise, dim=1)
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                ids_keep = ids_shuffle[:, :len_keep]

                chr_sentence_masked = torch.gather(chr_sentence, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
                mask_tokens = self.mask_token.expand(N, L - len_keep, D)
                x_ = torch.cat([chr_sentence_masked, mask_tokens], dim=1)
                chr_sentence = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

            token_sequence_parts.append(chr_sentence)

        token_sequence = torch.cat(token_sequence_parts, dim=1)

        # Build or refresh cache if needed (layout fixed -> only first time)
        self._build_layout_cache(lens_per_chr_per_scale, device=device)

        # If layout unexpectedly changed (e.g. h changed), cache rebuild happened.
        # Now apply vectorized embeddings / cached attention mask
        token_sequence = token_sequence + self.pos_embed(self._pos_index).unsqueeze(0)
        token_sequence = token_sequence + self.chr_embed(self._chr_index).unsqueeze(0)
        token_sequence = token_sequence + (self.size_embed(self._size_index) * self._size_mask_f.unsqueeze(1)).unsqueeze(0)

        attn_mask = self._attn_mask
        for blk in self.transformer_blocks:
            token_sequence = blk(token_sequence, attn_mask=attn_mask)

        return token_sequence, attn_mask, all_patches, self._token_class

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------
    def forward_loss(self, imgs, all_patches, x, token_class):
        device = x.device

        flag = 0
        all_loss = 0.0
        all_pred_chrs = []
        count = 0

        for chr_index, chr_list in enumerate(all_patches):
            all_pred_chrs.append(self.chr_decoder(x[:, flag : flag + 1, :]))
            flag += 1
            for size_index, size_list in enumerate(chr_list):
                l = size_list.shape[1]
                inputs = x[:, flag : flag + l, :]
                layer = self.block_decoder[size_index]
                preds = layer(inputs)

                L_tensor = size_list        # 已经是 [B,k,C,p,p]
                p = size_list.shape[-1]

                if self.enorm:
                    preds_reshaped = preds.view(preds.shape[0], preds.shape[1], p, p) * (
                        imgs[-1].unsqueeze(1).unsqueeze(2).repeat(1, preds.shape[1], p, p)
                    )
                else:
                    preds_reshaped = preds.view(preds.shape[0], preds.shape[1], p, p)

                L_tensor = L_tensor.squeeze(2)   # 去掉 channel
                if self.otherlf == "MSE":
                    loss = torch.sum((preds_reshaped - L_tensor) ** 2)
                else:
                    L_tensor = L_tensor.view(preds.shape[0], preds.shape[1], p, p)
                    L_tensor = (L_tensor > 0).float()
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(preds_reshaped, L_tensor)

                count += preds_reshaped.numel()
                all_loss = all_loss + loss
                flag += l

        patch_loss = (all_loss / count) if (self.otherlf == "MSE") else all_loss

        all_chr_flatten = []
        chr_single_loss = 0.0
        target_size = (self.img_size, self.img_size)
        count2 = 0

        for chr_index, original_chrs in enumerate(imgs):
            if chr_index < len(imgs) - 2:
                if self.chr_single_loss:
                    interpolated = F.interpolate(original_chrs, size=target_size, mode="bilinear", align_corners=True)
                    interpolated = self.expand_tensor(interpolated)

                    if self.enorm:
                        pred = all_pred_chrs[chr_index].view(interpolated.shape[0], interpolated.shape[1], -1) * imgs[-1]
                    else:
                        pred = all_pred_chrs[chr_index].view(interpolated.shape[0], interpolated.shape[1], -1)

                    if self.otherlf == "MSE":
                        loss = torch.sum((interpolated - pred) ** 2)
                    else:
                        interpolated = (interpolated > 0).float()
                        criterion = nn.BCEWithLogitsLoss()
                        loss = criterion(pred, interpolated)

                    chr_single_loss = chr_single_loss + loss

                count2 += all_pred_chrs[chr_index].shape[0] * all_pred_chrs[chr_index].shape[1] * (self.img_size ** 2)
                all_chr_flatten.append(self.expand_tensor(original_chrs))

        if self.otherlf == "MSE":
            chr_single_loss = chr_single_loss / max(count2, 1)

        all_chr_pred_goal = torch.cat(all_chr_flatten, dim=2).view(all_chr_flatten[0].shape[0], -1)

        weight = torch.ones_like(all_chr_pred_goal, device=device)
        N, L = all_chr_pred_goal.shape

        sparsity = (all_chr_pred_goal == 0).sum(dim=1).float() / float(L)
        if self.weight_mode != "mask":
            for i in range(N):
                sparsity_factor = sparsity[i]

                zero_percentage = torch.abs((3 * sparsity_factor - 2) / sparsity_factor)

                zero_indices = (all_chr_pred_goal[i] == 0).nonzero(as_tuple=True)[0]
                num_zeros = zero_indices.size(0)

                num_to_zero = int(num_zeros * zero_percentage)

                if num_zeros > 0:
                    selected_indices = zero_indices[torch.randperm(num_zeros)[:num_to_zero]]
                    weight[i, selected_indices] = 0
        else:
            weight[all_chr_pred_goal == 0] = self.weight

        if bool(self._layout_ready.item()) and self._chr_cls_index.numel() > 0:
            chr_cls_tokens = x.index_select(1, self._chr_cls_index).reshape(x.shape[0], -1)
        else:
            chr_cls_index = torch.nonzero(token_class == 1, as_tuple=False).flatten()
            chr_cls_tokens = x.index_select(1, chr_cls_index).reshape(x.shape[0], -1)

        latent_embed = self.latent_layer(chr_cls_tokens)
        cell_embed = self.pred_hidden(latent_embed)

        if self.enorm:
            chr_pred = self.cell_decoder(cell_embed) * imgs[-1]
        else:
            chr_pred = self.cell_decoder(cell_embed)

        chr_loss = torch.mean((chr_pred - all_chr_pred_goal) ** 2 * weight)
        return chr_pred, cell_embed, patch_loss, chr_loss, chr_single_loss

    # -------------------------
    # Pretrain functions (kept)
    # -------------------------
    def aepretrain(self, x):
        imgs = x
        device = next(self.parameters()).device

        token_sequence_parts = []
        all_pred_chrs = []
        for i in range(len(x) - 2):
            chr_cls = x[-2][:, i : i + 1, :]
            all_pred_chrs.append(self.chr_decoder(chr_cls))
            token_sequence_parts.append(chr_cls)
        token_sequence = torch.cat(token_sequence_parts, dim=1)

        all_chr_flatten = []
        chr_single_loss = 0.0
        target_size = (self.img_size, self.img_size)
        count = 0

        for chr_index, original_chrs in enumerate(imgs):
            if chr_index < len(imgs) - 2:
                if self.chr_single_loss:
                    interpolated = F.interpolate(original_chrs, size=target_size, mode="bilinear", align_corners=True)
                    interpolated = self.expand_tensor(interpolated)

                    if self.enorm:
                        pred = all_pred_chrs[chr_index].view(interpolated.shape[0], interpolated.shape[1], -1) * imgs[-1]
                    else:
                        pred = all_pred_chrs[chr_index].view(interpolated.shape[0], interpolated.shape[1], -1)

                    if self.otherlf == "MSE":
                        loss = torch.sum((interpolated - pred) ** 2)
                    else:
                        interpolated = (interpolated > 0).float()
                        criterion = nn.BCEWithLogitsLoss()
                        loss = criterion(pred, interpolated)

                    chr_single_loss = chr_single_loss + loss

                count += all_pred_chrs[chr_index].shape[0] * all_pred_chrs[chr_index].shape[1] * (self.img_size ** 2)
                all_chr_flatten.append(self.expand_tensor(original_chrs))

        if self.otherlf == "MSE":
            chr_single_loss = chr_single_loss / max(count, 1)

        all_chr_pred_goal = torch.cat(all_chr_flatten, dim=2).view(all_chr_flatten[0].shape[0], -1)

        weight = torch.ones_like(all_chr_pred_goal, device=device)
        weight[all_chr_pred_goal == 0] = self.weight

        chr_cls_tokens = token_sequence.reshape(token_sequence.shape[0], -1)
        latent_embed = self.latent_layer(chr_cls_tokens)
        cell_embed = self.pred_hidden(latent_embed)

        if self.enorm:
            chr_pred = self.cell_decoder(cell_embed) * imgs[-1]
        else:
            chr_pred = self.cell_decoder(cell_embed)

        chr_loss = torch.mean((chr_pred - all_chr_pred_goal) ** 2 * weight)
        return chr_loss, chr_single_loss

    def patch_pretrain(self, x):
        imgs = x
        device = next(self.parameters()).device

        all_patches = []
        token_sequence_parts = []

        chr_cls = torch.rand(x[0].shape[0], self.embed_dim, device=device).unsqueeze(1)

        for i in range(len(x) - 2):
            chr_patches = []
            input_image = x[i]
            _, _, h, _ = input_image.shape

            token_sequence_parts.append(chr_cls)
            for j, embed_layer in enumerate(self.multi_encoder):
                if h >= self.patch_size[j]:
                    tokens, diagonal_patches = embed_layer(input_image)
                    token_sequence_parts.append(tokens)
                    chr_patches.append(diagonal_patches)

            all_patches.append(chr_patches)

        x_tokens = torch.cat(token_sequence_parts, dim=1)

        flag = 0
        all_loss = 0.0
        count = 0

        for chr_index, chr_list in enumerate(all_patches):
            flag += 1
            for size_index, size_list in enumerate(chr_list):
                l = size_list.shape[1]
                inputs = x_tokens[:, flag : flag + l, :]
                layer = self.block_decoder[size_index]
                preds = layer(inputs)


                L_tensor = size_list
                p = size_list.shape[-1]

                if self.enorm:
                    preds_reshaped = preds.view(preds.shape[0], preds.shape[1], p, p) * (
                        imgs[-1].unsqueeze(1).unsqueeze(2).repeat(1, preds.shape[1], p, p)
                    )
                else:
                    preds_reshaped = preds.view(preds.shape[0], preds.shape[1], p, p)

                L_tensor = L_tensor.squeeze(2)

                if self.otherlf == "MSE":
                    loss = torch.sum((preds_reshaped - L_tensor) ** 2)
                else:
                    L_tensor = L_tensor.view(preds.shape[0], preds.shape[1], p, p)
                    L_tensor = (L_tensor > 0).float()
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(preds_reshaped, L_tensor)

                count += preds_reshaped.numel()
                all_loss = all_loss + loss
                flag += l

        patch_loss = (all_loss / count) if (self.otherlf == "MSE") else all_loss
        return patch_loss

    # -------------------------
    # Original forward (kept)
    # -------------------------
    def forward(self, imgs, mask_ratio=0.0):
        x, attn_mask, all_patches, token_class = self.forward_encoder(imgs, mask_ratio=mask_ratio)
        chr_pred, cell_embed, patch_loss, chr_loss, chr_single_loss = self.forward_loss(imgs, all_patches, x, token_class)
        return x, chr_pred, cell_embed, token_class, patch_loss, chr_loss, chr_single_loss

    # ---------------------------------------------------------
    # Highest-performance inference
    # ---------------------------------------------------------
    @torch.inference_mode()
    def inference(self, imgs, *, return_tokens=False, mode="full"):
        """
        Inference modes
        ---------------
        mode="full":
            normal inference
            imgs -> encoder -> transformer -> chr cls tokens -> cell embedding

        mode="pretrain":
            pretrain-style inference
            directly use imgs[-2] as chromosome cls tokens
            no transformer involved

        Returns
        -------
        if return_tokens:
            full mode    -> (x, chr_pred, cell_embed, token_class)
            pretrain mode-> (token_sequence, chr_pred, cell_embed, None)
        else:
            (chr_pred, cell_embed)
        """

        if mode == "pretrain":
            # directly use chromosome cls tokens, no transformer
            token_sequence_parts = []
            for i in range(len(imgs) - 2):
                chr_cls = imgs[-2][:, i : i + 1, :]   # [B,1,D]
                token_sequence_parts.append(chr_cls)

            token_sequence = torch.cat(token_sequence_parts, dim=1)   # [B, chr_num, D]
            chr_cls_tokens = token_sequence.reshape(token_sequence.shape[0], -1)

            latent_embed = self.latent_layer(chr_cls_tokens)
            cell_embed = self.pred_hidden(latent_embed)

            if self.enorm:
                chr_pred = self.cell_decoder(cell_embed) * imgs[-1]
            else:
                chr_pred = self.cell_decoder(cell_embed)

            if return_tokens:
                return token_sequence, chr_pred, cell_embed, None
            return chr_pred, cell_embed

        elif mode == "full":
            x, _, _, token_class = self.forward_encoder(imgs, mask_ratio=0.0)

            # use cached chr_cls indices if available
            if bool(self._layout_ready.item()) and self._chr_cls_index.numel() > 0:
                chr_cls_tokens = x.index_select(1, self._chr_cls_index).reshape(x.shape[0], -1)
            else:
                chr_cls_index = torch.nonzero(token_class == 1, as_tuple=False).flatten()
                chr_cls_tokens = x.index_select(1, chr_cls_index).reshape(x.shape[0], -1)

            latent_embed = self.latent_layer(chr_cls_tokens)
            cell_embed = self.pred_hidden(latent_embed)

            if self.enorm:
                chr_pred = self.cell_decoder(cell_embed) * imgs[-1]
            else:
                chr_pred = self.cell_decoder(cell_embed)

            if return_tokens:
                return x, chr_pred, cell_embed, token_class
            return chr_pred, cell_embed

        else:
            raise ValueError(f"Unsupported inference mode: {mode}")