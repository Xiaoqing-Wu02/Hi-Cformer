
import random

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint

from torch.jit import Final
from torch.utils.data import TensorDataset, DataLoader


from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from timm.layers import Mlp, DropPath, use_fused_attn
from .cluster import run_leiden
import scanpy as sc



def BandNorm(onechr_pic, ifdiagonal=False):
    """
    Applies BandNorm normalization to a list of single-cell Hi-C contact matrices.

    This function implements a band-wise normalization strategy inspired by the method
    described in Zheng et al. (2022): "Normalization and de-noising of single-cell Hi-C data 
    with BandNorm and scVI-3D", Genome Biology, 23(1), 222.

    Each matrix in the input list is normalized along its diagonal bands (offsets), reducing
    cell-specific coverage biases. Optionally, the main diagonal can also be normalized.
    After normalization, matrices are symmetrized to ensure they remain valid contact maps.

    Parameters:
        onechr_pic (list of np.ndarray): A list of square 2D contact matrices, one per cell.
        ifdiagonal (bool): Whether to include the main diagonal (offset=0) in the normalization.

    Returns:
        list of np.ndarray: The normalized and symmetrized contact matrices.
    """
    D,_ = onechr_pic[0].shape
    for i in range(0, D):
        if ifdiagonal == True:
            sums = 0
            for item in onechr_pic:
                diagonal_values = np.diagonal(item, offset=i)
                diagonal_sum = np.sum(diagonal_values)
                sums = diagonal_sum + sums
                diagonal = []
                for j in range(0, D-i):
                    diagonal.append([j, j + i])
                diagonal = np.array(diagonal)
                if diagonal_sum != 0:
                    item[diagonal[:,0], diagonal[:,1]] = item[diagonal[:,0], diagonal[:,1]]/diagonal_sum
            sums = sums/len(onechr_pic)
            for item in onechr_pic:
                diagonal = []
                for j in range(0, D-i):
                    diagonal.append([j, j + i])
                diagonal = np.array(diagonal)
                item[diagonal[:,0], diagonal[:,1]] = item[diagonal[:,0], diagonal[:,1]]*sums
                
        elif i!=0:
            sums = 0
            for item in onechr_pic:
                diagonal_values = np.diagonal(item, offset=i)
                diagonal_sum = np.sum(diagonal_values)
                sums = diagonal_sum + sums
                diagonal = []
                for j in range(0, D-i):
                    diagonal.append([j, j + i])
                diagonal = np.array(diagonal)
                if diagonal_sum != 0:
                    item[diagonal[:,0], diagonal[:,1]] = item[diagonal[:,0], diagonal[:,1]]/diagonal_sum
            sums = sums/len(onechr_pic)
            for item in onechr_pic:
                diagonal = []
                for j in range(0, D-i):
                    diagonal.append([j, j + i])
                diagonal = np.array(diagonal)
                item[diagonal[:,0], diagonal[:,1]] = item[diagonal[:,0], diagonal[:,1]]*sums
                
    for item in onechr_pic:
        indices = np.tril_indices(D, k=-1)
        item[indices[0],indices[1]] = item.T[indices[0],indices[1]]
        
    return onechr_pic



class Attention(nn.Module):
    """
    Multi-head self-attention module with optional attention masking support.

    This module is based on the Attention implementation from the timm library, with
    an extension to support custom attention masks. It supports both fused attention
    (via PyTorch's scaled_dot_product_attention) and the standard scaled dot-product
    attention mechanism.

    If `attn_mask` is provided, it will be added to the attention logits before the
    softmax operation, allowing for flexible masking strategies (e.g., padding masks,
    causal masks, etc.).

    Parameters:
        dim (int): Total input dimension (must be divisible by num_heads).
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, adds a learnable bias to q, k, v projections.
        qk_norm (bool): If True, applies normalization to q and k vectors.
        attn_drop (float): Dropout rate on attention weights.
        proj_drop (float): Dropout rate on output projection.
        norm_layer (nn.Module): Normalization layer to apply to q and k (e.g., LayerNorm).

    Inputs:
        x (Tensor): Input tensor of shape (B, N, C), where B is the batch size,
                    N is the sequence length, and C is the input dimension.
        attn_mask (Tensor, optional): Attention mask tensor of shape broadcastable to
                    (B * num_heads, N, N), or None. Defaults to None.

    Returns:
        Tensor: Output tensor of shape (B, N, C), same as input shape.
    """

    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def custom_attention(self, query, key, value, attn_mask):
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = query * self.scale
            attn = q @ key.transpose(-2, -1)
            if attn_mask != None:
                attn = attn + attn_mask  # Add the attention mask here
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ value
        return x

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = self.custom_attention(q, k, v, attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def cal_expand_tensor(img):
    B, C, H, _ = img.shape
    L = int(H * (H + 1) // 2)
    return L

class LayerScale(nn.Module):
    '''
    code from the timm library
    '''
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class BlockWithCustomMask(nn.Module):
    """
    Transformer block adapted from the timm library's implementation, with support for a custom attention mask.

    This block consists of:
    - A multi-head self-attention module that optionally accepts an attention mask (`attn_mask`),
    allowing flexible masking of attention weights (e.g., for causal masking or sparsity).
    - Pre-attention Layer Normalization and optional LayerScale and stochastic depth (DropPath) regularization.
    - A feed-forward MLP module with its own normalization, LayerScale, and DropPath.
    
    Parameters such as `init_values` allow LayerScale initialization, and
    drop_path controls stochastic depth rate, consistent with timm design.

    The main extension here is the `attn_mask` argument propagated to the attention layer,
    enabling customized attention masking beyond the standard full attention.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class block_encoder(nn.Module):
    """
    Modified Patch Embedding module that extracts and processes only the diagonal patches from the input feature map.

    Unlike a standard patch embedding which extracts all patches in a grid,
    this module extracts patches along the main diagonal of the input tensor,
    applies a convolutional projection to each diagonal patch separately,
    then concatenates and optionally flattens the results.

    Returns both:
    - The embedded patches tensor (after projection, normalization, and optional flattening)
    - The list of extracted diagonal patches before projection, allowing access to the original diagonal regions.

    Parameters:
    - patch_size: Size of each square patch.
    - in_chans: Number of input channels.
    - embed_dim: Dimension of the embedding output.
    - stride: Step size between patches.
    - padding: Padding applied to input tensor.
    - norm_layer: Optional normalization layer applied after embedding.
    - flatten: Whether to flatten the output to (batch, num_patches, embed_dim).
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, stride=16, padding=0, norm_layer=None, flatten=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.stride = stride
        self.padding = padding
        self.flatten = flatten
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (0, self.padding, 0, self.padding), value=0)
        k = (H + self.padding - self.patch_size)//self.stride + 1
        diagonal_patches = []
        for m in range(0, k*self.stride, self.stride):
            diagonal_patches.append(x[:,:,m:m+self.patch_size,m:m+self.patch_size])
        result = []
        for item in diagonal_patches:
            result.append(self.proj(item))
        x = torch.cat(result, dim=2)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, diagonal_patches
    
def calNMI(model1, data_loader_test, labels,device):
    model1.eval()
    cell_embed_matrix = []
    with torch.no_grad():
        for inputs in data_loader_test:
            result = []
            inputs = [item.to(device) for item in inputs]
            x, attn_mask, all_patches, token_class = model1.forward_encoder(inputs)
            chr_cls_index = np.where(token_class == 1)[0]
            chr_cls_tokens = x[:,chr_cls_index, :]
            chr_cls_tokens = chr_cls_tokens.view(chr_cls_tokens.shape[0], -1)
            cell_embed = model1.latent_layer(chr_cls_tokens)
            cell_embed = model1.pred_hidden(cell_embed)
            cell_embed = torch.nn.functional.gelu(cell_embed)
            cell_embed = np.squeeze(cell_embed.cpu().numpy())
            cell_embed_matrix.append(cell_embed)
    cell_embed_matrix = np.vstack(cell_embed_matrix)
    scaler = StandardScaler()
    cell_embed_standardized = scaler.fit_transform(cell_embed_matrix)

    nmi = []
    ari = []
    homo = []
    ami = []
    ATAC_pca = sc.AnnData(cell_embed_matrix,dtype = 'float32')
    ATAC_pca.obs['label'] = labels
    sc.pp.neighbors(ATAC_pca, n_neighbors=15, use_rep='X',random_state=int(0.1*1000))
    louvain_df = run_leiden(ATAC_pca,'label','cluster',seed=int(0.1*1000))
    nmi.append(louvain_df['NMI'][1])
    ari.append(louvain_df['ARI'][1])
    homo.append(louvain_df['Homo'][1])
    ami.append(louvain_df['AMI'][1])
    ATAC_pca = sc.AnnData(cell_embed_standardized,dtype = 'float32')
    ATAC_pca.obs['label'] = labels
    sc.pp.neighbors(ATAC_pca, n_neighbors=15, use_rep='X',random_state=int(0.1*1000))
    louvain_df = run_leiden(ATAC_pca,'label','cluster',seed=int(0.1*1000))
    nmi.append(louvain_df['NMI'][1])
    ari.append(louvain_df['ARI'][1])
    homo.append(louvain_df['Homo'][1])
    ami.append(louvain_df['AMI'][1])
    return nmi, ari,homo,ami
def calNMI_pretrain(model1, data_loader_test, labels,device):
    model1.eval()
    cell_embed_matrix = []
    with torch.no_grad():
        for inputs in data_loader_test:
            result = []
            inputs = [item.to(device) for item in inputs]
            token_sequence = []#所有token
            imgs = inputs
            x = inputs
            for i in range(len(x)-2):
                chr_cls = x[len(x)-2][:,i:i+1,:]
                token_sequence.append(chr_cls)#此时插入chr_cls_token
            token_sequence = torch.cat(token_sequence, dim=1)
            token_sequence = model1.norm(token_sequence)
            x = token_sequence
            chr_cls_tokens = x
            chr_cls_tokens = chr_cls_tokens.view(chr_cls_tokens.shape[0], -1)
            latent_embed = model1.latent_layer(chr_cls_tokens)
            cell_embed = model1.pred_hidden(latent_embed)
            cell_embed = torch.nn.functional.gelu(cell_embed)
            cell_embed = np.squeeze(cell_embed.cpu().numpy())
            cell_embed_matrix.append(cell_embed)
    cell_embed_matrix = np.vstack(cell_embed_matrix)
    scaler = StandardScaler()
    cell_embed_standardized = scaler.fit_transform(cell_embed_matrix)

    nmi = []
    ari = []
    homo = []
    ami = []
    ATAC_pca = sc.AnnData(cell_embed_matrix,dtype = 'float32')
    ATAC_pca.obs['label'] = labels
    sc.pp.neighbors(ATAC_pca, n_neighbors=15, use_rep='X',random_state=int(0.1*1000))
    louvain_df = run_leiden(ATAC_pca,'label','cluster',seed=int(0.1*1000))
    nmi.append(louvain_df['NMI'][1])
    ari.append(louvain_df['ARI'][1])
    homo.append(louvain_df['Homo'][1])
    ami.append(louvain_df['AMI'][1])
    ATAC_pca = sc.AnnData(cell_embed_standardized,dtype = 'float32')
    ATAC_pca.obs['label'] = labels
    sc.pp.neighbors(ATAC_pca, n_neighbors=15, use_rep='X',random_state=int(0.1*1000))
    louvain_df = run_leiden(ATAC_pca,'label','cluster',seed=int(0.1*1000))
    nmi.append(louvain_df['NMI'][1])
    ari.append(louvain_df['ARI'][1])
    homo.append(louvain_df['Homo'][1])
    ami.append(louvain_df['AMI'][1])
    return nmi, ari,homo,ami
def cal_expand_tensor(img):
    B, C, H, _ = img.shape
    L = int(H * (H + 1) // 2)  
    return L

def get_embeddings(model1, data_loader_test, labels,device):
    model1.eval()
    cell_embed_matrix = []
    with torch.no_grad():
        for inputs in data_loader_test:
            result = []
            inputs = [item.to(device) for item in inputs]
            x, attn_mask, all_patches, token_class = model1.forward_encoder(inputs)
            chr_cls_index = np.where(token_class == 1)[0]
            chr_cls_tokens = x[:,chr_cls_index, :]
            chr_cls_tokens = chr_cls_tokens.view(chr_cls_tokens.shape[0], -1)
            cell_embed = model1.latent_layer(chr_cls_tokens)
            cell_embed = model1.pred_hidden(cell_embed)
            cell_embed = torch.nn.functional.gelu(cell_embed)
            cell_embed = np.squeeze(cell_embed.cpu().numpy())
            cell_embed_matrix.append(cell_embed)
    cell_embed_matrix = np.vstack(cell_embed_matrix)
    return cell_embed_matrix

def get_imputation(model1, data_loader_test, labels,device):
    model1.eval()
    chr_pred_matrix = []
    with torch.no_grad():
        for inputs in data_loader_test:
            result = []
            inputs = [item.to(device) for item in inputs]
            x, attn_mask, all_patches, token_class = model1.forward_encoder(inputs)
            chr_cls_index = np.where(token_class == 1)[0]
            chr_cls_tokens = x[:,chr_cls_index, :]
            chr_cls_tokens = chr_cls_tokens.view(chr_cls_tokens.shape[0], -1)
            cell_embed = model1.latent_layer(chr_cls_tokens)
            cell_embed = model1.pred_hidden(cell_embed)
            if hasattr(model1, 'enorm') and model1.enorm == True:
                chr_pred = model1.whole_pred(torch.nn.functional.gelu(cell_embed))*inputs[-1]
            else:
                chr_pred = model1.whole_pred(torch.nn.functional.gelu(cell_embed))

            
            chr_pred = np.squeeze(chr_pred.cpu().numpy())
            chr_pred_matrix.append(chr_pred)
        chr_pred_matrix = np.vstack(chr_pred_matrix)
    return chr_pred_matrix

def get_imputation_nbr(model1, embeddings, device, all_pics, nbr_num=4):
    model1.eval()
    chr_pred_matrix = []
    similarity_matrix = cosine_similarity(embeddings)
    sorted_indices = np.argsort(similarity_matrix, axis=1)[:, -(nbr_num+1):]
    neighbor_similarities = np.take_along_axis(similarity_matrix, sorted_indices, axis=1)
    weighted_sum = np.sum(embeddings[sorted_indices] * neighbor_similarities[..., np.newaxis], axis=1)
    sum_of_weights = np.sum(neighbor_similarities, axis=1)
    embeddings = weighted_sum / sum_of_weights[:, np.newaxis]
    new_embeddings = torch.from_numpy(embeddings)
    dataset = TensorDataset(*[new_embeddings, all_pics[-1]])
    data_loader_test = DataLoader(dataset, batch_size=16, shuffle=False)

    with torch.no_grad():
        for inputs in data_loader_test:
            inputs = [item.to(device) for item in inputs]
            cell_embed = inputs[0]
            if hasattr(model1, 'enorm') and model1.enorm == True:
                chr_pred = model1.whole_pred(cell_embed)*inputs[-1]
            else:
                chr_pred = model1.whole_pred(cell_embed)

            
            chr_pred = np.squeeze(chr_pred.cpu().numpy())
            chr_pred_matrix.append(chr_pred)
        chr_pred_matrix = np.vstack(chr_pred_matrix)
    
    return chr_pred_matrix, embeddings

import torch
from torch.utils.data import TensorDataset, random_split
import random

def sample_dataset(dataset, ratio, seed=1):

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    total_size = len(dataset)
    sample_size = int(total_size * ratio)
    
    indices = random.sample(range(total_size), sample_size)
    
    sampled_data = torch.utils.data.Subset(dataset, indices)
    return sampled_data, indices

def sample_by_label(dataset, labels, target_label, ratio, seed=1):

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)


    target_indices = [i for i, label in enumerate(labels) if label == target_label]
    

    target_sample_size = int(len(target_indices) * ratio)
    
   
    sampled_target_indices = random.sample(target_indices, target_sample_size)
    

    other_indices = [i for i, label in enumerate(labels) if label != target_label]
    
    
    final_indices = sampled_target_indices + other_indices
    

    sampled_data = torch.utils.data.Subset(dataset, final_indices)
    sampled_labels = [labels[i] for i in final_indices]
    
    return sampled_data, sampled_labels

def apply_max_distance(dataset, max_distance):

    def apply_zeroing(tensor, max_distance):
  

        shape = tensor.shape
        if len(shape) < 2:
            return tensor 
        

        x_dim, y_dim = shape[-2], shape[-1]
        
        for i in range(x_dim):
            for j in range(y_dim):
                if abs(i - j) > max_distance:
                    tensor[..., i, j] = 0
        return tensor

    modified_data = []
    
    for tensor in dataset.tensors:
        modified_data.append(apply_zeroing(tensor, max_distance))
    
    return TensorDataset(*modified_data)
