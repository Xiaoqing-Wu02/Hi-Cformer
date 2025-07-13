import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from .utils import block_encoder, BlockWithCustomMask


class hicformer(nn.Module):
    def __init__(self, whole_size=1000, img_size=112, hidden_dim=60, patch_size=[8, 16, 32, 64, 128], encoder_dim=[], chr_num=21, in_chans=1,
                 embed_dim=128, depth=4, num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, chr_mutual_visibility=True, weight=0.1, chr_single_loss=False, otherlf='MSE', enorm=False, num_patches=24, weight_mode='mask'):
        super().__init__()

        self.chr_single_loss = chr_single_loss
        self.embed_dim = embed_dim
        self.chr_mutual_visibility = chr_mutual_visibility
        self.patch_size = patch_size
        # The total length of concatenated upper-triangular flattened contact maps of all chromosomes as input
        self.whole_size = whole_size
        self.multi_encoder = nn.ModuleList([block_encoder(s, in_chans, embed_dim, stride=s, padding=0, norm_layer=norm_layer) for s in patch_size])
        self.chr_num = chr_num
        self.img_size = img_size
        self.weight = weight
        # This parameter only relates to position embeddings, thus set as the number of patches when patch size is 2
        self.num_patches = num_patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Embedding layer including cell/chromosome cls token embeddings
        self.pos_embed = nn.Embedding(num_embeddings=num_patches + 1, embedding_dim=embed_dim)
        self.size_embed = nn.Embedding(num_embeddings=len(self.multi_encoder), embedding_dim=embed_dim)
        self.chr_embed = nn.Embedding(num_embeddings=chr_num, embedding_dim=embed_dim)
        self.transformer_blocks = nn.ModuleList([
            BlockWithCustomMask(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.otherlf = otherlf
        self.enorm = enorm
        self.weight_mode = weight_mode
        
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
        self.chr_decoder = nn.Sequential(nn.Linear(embed_dim, int(img_size*(img_size+1)/2), bias=True), nn.ReLU())
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        torch.nn.init.xavier_uniform_(self.pos_embed.weight)
        torch.nn.init.xavier_uniform_(self.size_embed.weight)
        torch.nn.init.xavier_uniform_(self.chr_embed.weight)
        torch.nn.init.normal_(self.mask_token, std=.02)
        for i in range(len(self.multi_encoder)):
            w = self.multi_encoder[i].proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    


    def forward_encoder(self, x, mask_ratio=0.0):
    # x is a list of length chr_num, each element shape: [batch, channel, h, w]
    # The second last element is for cls token, shape: [batch, chr_cum, embedding]
    # Lengths to save: total tokens per chromosome, tokens per scale within each chromosome
        all_patches = []  # Nested list to save diagonal blocks: length chr_num, each contains 4 lists (one per patch size),
                        # smallest element shape: (B, C, H, W)
        token_sequence = []  # All tokens combined
        device = next(self.parameters()).device
        replace_mask = []  # 0 means not replaced, 1 means replaced

        for i in range(len(x) - 2):
            chr_patches = []  # List to save all size patches in one chromosome
            chr_sentence = []
            input_image = x[i]
            # interpolated_tensor = F.interpolate(input_image, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
            chr_cls = x[len(x) - 2][:, i:i+1, :]
            chr_sentence.append(chr_cls)
            # token_sequence.append(chr_cls) # Insert chr_cls token here

            for j in range(len(self.multi_encoder)):
                input_image = x[i]
                _, _, h, _ = input_image.shape
                if h >= self.patch_size[j]:
                    embed_layer = self.multi_encoder[j]
                    tokens, diagonal_patches = embed_layer(input_image)
                    chr_sentence.append(tokens)
                    # token_sequence.append(tokens)
                    chr_patches.append(diagonal_patches)
            all_patches.append(chr_patches)

            chr_sentence = torch.cat(chr_sentence, dim=1)
            chr_len = chr_sentence.shape[1]
            N, L, D = chr_sentence.shape  # batch, length, dim
            len_keep = int(L * (1 - mask_ratio))

            noise = torch.rand(N, L, device=device)  # noise in [0, 1]

            # Sort noise for each sample: smaller noise means keep, larger means remove
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # Keep the first subset (lowest noise)
            ids_keep = ids_shuffle[:, :len_keep]
            chr_sentence_masked = torch.gather(chr_sentence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # Generate binary mask: 0 = keep, 1 = remove
            mask_replace = torch.ones([N, L], device=chr_sentence.device)
            mask_replace[:, :len_keep] = 0
            # Unshuffle to get mask aligned with original sequence
            mask_replace = torch.gather(mask_replace, dim=1, index=ids_restore)
            mask_tokens = self.mask_token.repeat(chr_sentence_masked.shape[0], ids_restore.shape[1] - chr_sentence_masked.shape[1], 1)
            x_ = torch.cat([chr_sentence_masked[:, :, :], mask_tokens], dim=1)  # no cls token
            # Unshuffle tokens to original order
            chr_sentence = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, chr_sentence_masked.shape[2]))
            replace_mask.append(mask_replace)
            token_sequence.append(chr_sentence)

        replace_mask = torch.cat(replace_mask, dim=1)
        token_sequence = torch.cat(token_sequence, dim=1)
        flag = 1
        all_chr_lens = []
        token_class = torch.zeros(token_sequence.shape[1] + 1)
        token_class[0] = -1  # -1 means cell cls token

        for chr_index, chr_list in enumerate(all_patches):
            chr_len = 0
            input_index = torch.tensor([0, ], dtype=torch.long).to(device)
            embedded_vector = self.pos_embed(input_index)
            token_sequence[:, flag - 1:flag, :] = token_sequence[:, flag - 1:flag, :] + embedded_vector.unsqueeze(0)
            token_class[flag] = 1  # 1 means chromosome cls token

            for size_index, size_list in enumerate(chr_list):
                l = len(size_list)
                input_index = torch.tensor(list(range(1, 1 + l)), dtype=torch.long).to(device)
                embedded_vector = self.pos_embed(input_index)
                token_sequence[:, flag:flag + l, :] = token_sequence[:, flag:flag + l, :] + embedded_vector.unsqueeze(0)

                input_index = torch.tensor([size_index], dtype=torch.long).to(device)
                embedded_vector = self.size_embed(input_index)
                token_sequence[:, flag:flag + l, :] = token_sequence[:, flag:flag + l, :] + embedded_vector.unsqueeze(0)

                chr_len += l
                flag += l
            all_chr_lens.append(chr_len + 1)

            input_index = torch.tensor([chr_index], dtype=torch.long).to(device)
            embedded_vector = self.chr_embed(input_index)
            token_sequence[:, flag - chr_len - 1:flag, :] = token_sequence[:, flag - chr_len - 1:flag, :] + embedded_vector.unsqueeze(0)
            flag += 1

        L = token_sequence.shape[1]
        attn_mask = torch.full((L, L), float('-inf'), dtype=torch.float32)
        flag = 0
        m = []
        for l in all_chr_lens:
            m.append(flag)
            attn_mask[flag:flag + l, flag:flag + l] = 0.0
            flag += l

        if self.chr_mutual_visibility:
            for i in m:
                for j in m:
                    attn_mask[i, j] = 0.0
                    attn_mask[j, i] = 0.0

        # Apply Transformer blocks
        N, L = replace_mask.shape
        attn_mask = attn_mask.to(device)
        for blk in self.transformer_blocks:
            token_sequence = blk(token_sequence, attn_mask=attn_mask)

        token_class = token_class[1:]
        return token_sequence, attn_mask, all_patches, token_class


    def expand_tensor(self, img):
        B, C, H, _ = img.shape
        L = int(H * (H + 1) // 2)  # Calculate the number of elements in the upper triangular matrix
        img = img.view(B, C, -1)  # Flatten the matrix into one dimension

        # Create upper triangular mask
        mask = torch.zeros(H, H)
        for i in range(H):
            for j in range(i, H):
                mask[i, j] = 1

        img = img[:, :, mask.view(-1).bool()]  # Keep only the upper triangular part using the mask
        return img

    def forward_loss(self, imgs, all_patches, x, token_class):
        """
        Compute the forward loss for the model.
        
        Args:
            imgs: List of length chr_num, each element is a tensor of shape [batch, channel, height, width].
            all_patches: Nested list storing diagonal patches. It contains chr_num lists, each with 4 size-based lists,
                        where each smallest element has shape (B, C, H, W).
            x: Tensor of shape [N, L, D], including hidden layer outputs of cell_cls and chr_cls tokens.
            token_class: Token classification labels (used to identify chr_cls tokens).
        
        Returns:
            all_loss: Total combined loss.
            chr_pred: Chromosome-level predictions.
            cell_embed: Cell embedding after prediction head.
            patch_loss: Loss related to patch-level predictions.
            chr_loss: Loss related to chromosome-level predictions.
            chr_single_loss: Additional single chromosome loss if enabled.
        """
        device = next(self.parameters()).device
        flag = 0
        all_loss = 0
        all_pred_chrs = []
        count = 0
        for chr_index, chr_list in enumerate(all_patches):
            all_pred_chrs.append(self.chr_decoder(x[:, flag:flag+1, :]))
            flag = flag + 1
            for size_index, size_list in enumerate(chr_list):
                l = len(size_list)
                inputs = x[:, flag:flag+l, :]
                layer = self.block_decoder[size_index]
                preds = layer(inputs)
                L_tensor = torch.stack(size_list, dim=1)
                p = size_list[0].shape[2]
                if self.enorm == True:
                    preds_reshaped = preds.view(preds.shape[0], preds.shape[1], p, p) * (imgs[-1].unsqueeze(1).unsqueeze(2).repeat(1, preds.shape[1], p, p))
                else:
                    preds_reshaped = preds.view(preds.shape[0], preds.shape[1], p, p)
                # Calculate squared loss
                L_tensor = torch.squeeze(L_tensor)
                if self.otherlf == 'MSE':
                    loss = torch.sum((preds_reshaped - L_tensor) ** 2)
                else:
                    L_tensor = L_tensor.view(preds.shape[0], preds.shape[1], p, p)
                    quantile_98 = torch.quantile(L_tensor, 0.98)
                    L_tensor = (L_tensor > 0).float()
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(preds_reshaped, L_tensor)
                count = count + preds_reshaped.shape[0] * preds_reshaped.shape[1] * p * p
                all_loss = all_loss + loss
                flag = flag + l
        if self.otherlf == 'MSE':     
            patch_loss = all_loss / count
        else:
            patch_loss = all_loss
        all_chr_flatten = []
        target_size = (self.img_size, self.img_size)
        chr_single_loss =  0
        count = 0
        for chr_index, original_chrs in enumerate(imgs):
            if chr_index < len(imgs) - 2:
                if self.chr_single_loss:
                    interpolated_tensor = F.interpolate(original_chrs, size=target_size, mode='bilinear', align_corners=True)
                    interpolated_tensor = self.expand_tensor(interpolated_tensor)
                    if self.enorm == True:
                        tensor1_reshaped = all_pred_chrs[chr_index].view(interpolated_tensor.shape[0], interpolated_tensor.shape[1], -1) * imgs[-1]
                    else:
                        tensor1_reshaped = all_pred_chrs[chr_index].view(interpolated_tensor.shape[0], interpolated_tensor.shape[1], -1)
                    if self.otherlf == 'MSE':  
                        loss = torch.sum((interpolated_tensor - tensor1_reshaped) ** 2)
                    else:
                        #quantile_98 = torch.quantile(interpolated_tensor, 0.98)
                        interpolated_tensor = (interpolated_tensor > 0).float()
                        criterion = nn.BCEWithLogitsLoss()
                        loss = criterion(tensor1_reshaped, interpolated_tensor)
                    chr_single_loss = chr_single_loss + loss
                count = count + all_pred_chrs[chr_index].shape[0] * all_pred_chrs[chr_index].shape[1] * self.img_size ** 2
                flatten_tensor = self.expand_tensor(original_chrs)
                all_chr_flatten.append(flatten_tensor)
        if self.otherlf == 'MSE':  
            chr_single_loss = chr_single_loss / count
        else:
            chr_single_loss = chr_single_loss
        all_chr_pred_goal = torch.cat(all_chr_flatten, dim=2)
        all_chr_pred_goal = all_chr_pred_goal.view(all_chr_pred_goal.shape[0], -1)
        weight = torch.ones(all_chr_pred_goal.shape)
        N, L = all_chr_pred_goal.shape
        # Calculate sparsity: ratio of zeros in each row
        sparsity = (all_chr_pred_goal == 0).sum(dim=1).float() / L  # ratio of zeros per row
        # Create weight tensor
        weight = torch.ones_like(all_chr_pred_goal)
        # Adjust weights according to sparsity
        # If not using mask mode, randomly zero some weights based on sparsity factor
        if self.weight_mode != 'mask':
            for i in range(N):
                sparsity_factor = sparsity[i]  # can be adjusted as needed
                zero_percentage = torch.abs((3 * sparsity_factor - 2) / sparsity_factor)
                zero_indices = (all_chr_pred_goal[i] == 0).nonzero(as_tuple=True)[0]  # indices of zeros
                num_zeros = zero_indices.size(0)
                num_to_zero = int(num_zeros * zero_percentage)  # number to zero out
                if num_zeros > 0:
                    # Randomly select indices to zero weight
                    selected_indices = zero_indices[torch.randperm(num_zeros)[:num_to_zero]]
                    # Set selected weights to zero
                    weight[i, selected_indices] = 0
        else:
            weight[all_chr_pred_goal == 0] = self.weight
        chr_cls_index = np.where(token_class == 1)[0]
        chr_cls_tokens = x[:, chr_cls_index, :]
        chr_cls_tokens = chr_cls_tokens.view(chr_cls_tokens.shape[0], -1)
        latent_embed = self.latent_layer(chr_cls_tokens)
        cell_embed = self.pred_hidden(latent_embed)
        if self.enorm == True:
            chr_pred = self.cell_decoder(torch.nn.functional.gelu(cell_embed)) * imgs[-1]
        else:
            chr_pred = self.cell_decoder(torch.nn.functional.gelu(cell_embed))
        weight = weight.to(device)
        chr_loss = torch.mean((chr_pred - all_chr_pred_goal) ** 2 * weight)
        
        return  chr_pred, cell_embed, patch_loss, chr_loss, chr_single_loss

    def aepretrain(self, x):
        imgs = x
        token_sequence = []  # All tokens
        device = next(self.parameters()).device
        token_sequence = []  # All tokens
        all_pred_chrs = []
        for i in range(len(x)-2):
            chr_cls = x[len(x)-2][:, i:i+1, :]
            all_pred_chrs.append(self.chr_decoder(chr_cls))
            token_sequence.append(chr_cls)  # Insert chr_cls token here
        token_sequence = torch.cat(token_sequence, dim=1)
        all_chr_flatten = []
        chr_single_loss = 0.0
        target_size = (self.img_size, self.img_size)
        count = 0
        for chr_index, original_chrs in enumerate(imgs):
            if chr_index < len(imgs) - 2:
                if self.chr_single_loss:
                    interpolated_tensor = F.interpolate(original_chrs, size=target_size, mode='bilinear', align_corners=True)
                    interpolated_tensor = self.expand_tensor(interpolated_tensor)
                    if self.enorm == True:
                        tensor1_reshaped = all_pred_chrs[chr_index].view(interpolated_tensor.shape[0], interpolated_tensor.shape[1], -1) * imgs[-1]
                    else:
                        tensor1_reshaped = all_pred_chrs[chr_index].view(interpolated_tensor.shape[0], interpolated_tensor.shape[1], -1)
                    if self.otherlf == 'MSE':
                        loss = torch.sum((interpolated_tensor - tensor1_reshaped) ** 2)
                    else:
                        # quantile_98 = torch.quantile(interpolated_tensor, 0.98)
                        interpolated_tensor = (interpolated_tensor > 0).float()
                        criterion = nn.BCEWithLogitsLoss()
                        loss = criterion(tensor1_reshaped, interpolated_tensor)
                    chr_single_loss = chr_single_loss + loss
                count = count + all_pred_chrs[chr_index].shape[0] * all_pred_chrs[chr_index].shape[1] * self.img_size ** 2
                flatten_tensor = self.expand_tensor(original_chrs)
                all_chr_flatten.append(flatten_tensor)
        if self.otherlf == 'MSE':
            chr_single_loss = chr_single_loss / count
        else:
            chr_single_loss = chr_single_loss
        all_chr_pred_goal = torch.cat(all_chr_flatten, dim=2)
        all_chr_pred_goal = all_chr_pred_goal.view(all_chr_pred_goal.shape[0], -1)
        weight = torch.ones(all_chr_pred_goal.shape)
        weight[all_chr_pred_goal == 0] = self.weight
        chr_cls_tokens = token_sequence.view(token_sequence.shape[0], -1)
        latent_embed = self.latent_layer(chr_cls_tokens)
        cell_embed = self.pred_hidden(latent_embed)
        if self.enorm == True:
            chr_pred = self.cell_decoder(torch.nn.functional.gelu(cell_embed)) * imgs[-1]
        else:
            chr_pred = self.cell_decoder(torch.nn.functional.gelu(cell_embed))
        weight = weight.to(device)
        chr_loss = torch.mean((chr_pred - all_chr_pred_goal) ** 2 * weight)
        return chr_loss, chr_single_loss

    
    def patch_pretrain(self, x):
        imgs = x
        all_patches = []  # Nested list to save diagonal patches; contains chr_num lists, each with 4 sizes (number of patches per size).
                        # The smallest element shape is (B, C, H, W).
        token_sequence = []  # List to hold all tokens
        device = next(self.parameters()).device
        chr_cls = torch.rand(x[0].shape[0], self.embed_dim)
        chr_cls = chr_cls.to(device)
        chr_cls = chr_cls.unsqueeze(1)
        for i in range(len(x) - 2):
            chr_patches = []  # List to save patches of all sizes within one chromosome
            input_image = x[i]
            token_sequence.append(chr_cls)  # Insert chromosome class token here
            for j in range(len(self.multi_encoder)):
                input_image = x[i]
                _, _, h, _ = input_image.shape
                if h >= self.patch_size[j]:
                    embed_layer = self.multi_encoder[j]
                    tokens, diagonal_patches = embed_layer(input_image)
                    token_sequence.append(tokens)
                    chr_patches.append(diagonal_patches)
            all_patches.append(chr_patches)
        token_sequence = torch.cat(token_sequence, dim=1)
        x = token_sequence
        flag = 0
        all_loss = 0
        count = 0
        for chr_index, chr_list in enumerate(all_patches):
            # all_pred_chrs.append(self.chr_decoder(x[:, flag:flag+1, :]))
            flag = flag + 1
            for size_index, size_list in enumerate(chr_list):
                l = len(size_list)
                inputs = x[:, flag:flag + l, :]
                layer = self.block_decoder[size_index]
                preds = layer(inputs)
                L_tensor = torch.stack(size_list, dim=1)
                p = size_list[0].shape[2]
                if self.enorm == True:
                    preds_reshaped = preds.view(preds.shape[0], preds.shape[1], p, p) * (imgs[-1].unsqueeze(1).unsqueeze(2).repeat(1, preds.shape[1], p, p))
                else:
                    preds_reshaped = preds.view(preds.shape[0], preds.shape[1], p, p)
                # Compute squared loss
                L_tensor = torch.squeeze(L_tensor)
                if self.otherlf == 'MSE':
                    loss = torch.sum((preds_reshaped - L_tensor) ** 2)
                else:
                    L_tensor = L_tensor.view(preds.shape[0], preds.shape[1], p, p)
                    #quantile_98 = torch.quantile(L_tensor, 0.98)
                    L_tensor = (L_tensor > 0).float()
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(preds_reshaped, L_tensor)
                count = count + preds_reshaped.shape[0] * preds_reshaped.shape[1] * p * p
                all_loss = all_loss + loss
                flag = flag + l
        if self.otherlf == 'MSE':
            patch_loss = all_loss / count
        else:
            patch_loss = all_loss

        return patch_loss


    def forward(self, imgs, mask_ratio=0.0):
        x, attn_mask, all_patches, token_class = self.forward_encoder(imgs, mask_ratio=mask_ratio)
        chr_pred, cell_embed, patch_loss, chr_loss, chr_single_loss = self.forward_loss(imgs, all_patches, x, token_class)
        return x, chr_pred, cell_embed,token_class, patch_loss, chr_loss, chr_single_loss
    

