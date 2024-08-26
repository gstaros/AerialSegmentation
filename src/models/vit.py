import torch.nn as nn
import torch

class PatchEmbedding(nn.Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            in_channels=3, 
            embedding_dim=768
        ):

        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.nun_patches = (image_size // patch_size) ** 2

        self.project = nn.Conv2d(
                in_channels,
                embedding_dim,
                kernel_size=patch_size,
                stride=patch_size,
        )

    def forward(self, x):
        # B - Batch size, D - Embedding Dim, N - number of patches
        x = self.project(x)  # (B, D, N ** 0.5, N ** 0.5)
        x = x.flatten(2)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        return x


class PositionalEmbeddings(nn.Module):
    def __init__(self, seq_len, embedding_dim):
        #seq_len = N (num_patches) in classification tasks it would be num_patches + 1 due to cls token. In segmentation it's not needed
        super(PositionalEmbeddings, self).__init__()
        self.embedding = nn.Parameter(torch.zeros(1, seq_len, embedding_dim))
        
    def forward(self, x):
        return x + self.embedding



class Attention(nn.Module):
    def __init__(
            self, 
            embedding_dim:int, 
            num_heads:int, 
            qkv_bias: bool = False, 
            attention_dropout: float = 0., 
            fc_out_dropout: float = 0.
        ):

        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** (- 1/2)

        assert (embedding_dim == self.num_heads * self.head_dim ), "Wrong embeding or head size. Embeding need to be divisible by heads without remainder"

        self.qkv_linear = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        self.fc_out_dropout = nn.Dropout(fc_out_dropout)
        

    def forward(self, x):
        B, N, C = x.shape

        assert (C == self.embedding_dim), "Patchified image embeding is not of the same size as defined in init function"

        qkv = self.qkv_linear(x) #B, N, D * 3 
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.permute(2, 0, 3, 1, 4) # 3, B, n_heads, N, head_dim 

        queries, keys, values = qkv[0], qkv[1], qkv[2]

        dot_prod = (queries @ keys.transpose(-2, -1)) * self.scale
        dot_prod = dot_prod.softmax(dim=-1)
        dot_prod = self.attention_dropout(dot_prod)

        attention = (dot_prod @ values).transpose(1, 2).reshape(B, N, C)
        attention = self.fc_out(attention)

        return self.fc_out_dropout(attention)


class MLPLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int,
            mlp_dropout: float = 0.
        ):

        super(MLPLayer, self).__init__()
        self.MLP_block = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(mlp_dropout)
        )

    def forward(self, x):
        return self.MLP_block(x)

class EncoderBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            hidden_dim: int,
            attention_dropout: float = 0., 
            fc_out_dropout: float = 0.,
            mlp_dropout: float = 0.
    ):
        
        super(EncoderBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.attention = Attention(embedding_dim, num_heads, attention_dropout, fc_out_dropout)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPLayer(embedding_dim, hidden_dim, mlp_dropout)

    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, 
            num_layers, 
            embedding_dim: int, 
            num_heads: int,
            hidden_dim: int,
            attention_dropout: float = 0., 
            fc_out_dropout: float = 0.,
            mlp_dropout: float = 0.
    ):

        super(Transformer, self).__init__()
        self.dropout = nn.Dropout(fc_out_dropout)
        self.norm = nn.LayerNorm(embedding_dim)

        self.blocks = nn.ModuleList(
            [EncoderBlock(embedding_dim, 
                            num_heads, 
                            hidden_dim, 
                            attention_dropout, 
                            fc_out_dropout,
                            mlp_dropout) for _ in range(num_layers)])
            
    def forward(self, x):
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)



class ConvSegmenter(nn.Module):
    def __init__(self, 
        in_layers,
        out_layers
    ):

        super(ConvSegmenter, self).__init__()
        self.segmenter = nn.Conv2d(in_layers, out_layers, 3, 1, 1)
        self.softmax = nn.Softmax(dim=1)
            
    def forward(self, x):
        x = self.segmenter(x)
        return self.softmax(x)
    


class PatchMerge(nn.Module):
    def __init__(self, 
                 image_size: int= 256, 
                 patch_size: int = 16, 
                 num_classes: int =3, 
                 embedding_dim: int= 768
        ):

        super(PatchMerge, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = embedding_dim

        self.deconv = nn.ConvTranspose2d(
            embedding_dim, 
            num_classes, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # B - Batch size, N - number of patches, D - Embedding Dim
        B, _, N, D = x.shape
        assert N == self.num_patches, "Number of patches does not match."
        
        # argmax to get only one prediction for class per pixel
        x = x.argmax(dim=1).squeeze(dim=1)

        # Reshape and transpose back to (B, D, sqrt(N), sqrt(N))
        x = x.transpose(1, 2)  # (B, D, N)
        x = x.view(B, D, int(self.image_size // self.patch_size), int(self.image_size // self.patch_size))

        # Apply the transposed convolution to reconstruct the image
        x = self.deconv(x.float())  # (B, C, H, W)
        return x


class ViT(nn.Module):
    def __init__(self,
                 image_size: int = 256,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 13,
                 embedding_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 hidden_dim: int = 3072,
                 attention_dropout: float = 0.,
                 fc_out_dropout: float = 0.,
                 mlp_dropout: float = 0.):

        super(ViT, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        seq_length = (image_size // patch_size) ** 2

        # Embedding creation
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embedding_dim)
        self.pos_embedding = PositionalEmbeddings(seq_length, embedding_dim)

        self.transformer = Transformer(num_layers,
                                       embedding_dim,
                                       num_heads,
                                       hidden_dim,
                                       attention_dropout,
                                       fc_out_dropout,
                                       mlp_dropout)
        
        # self.segmentation = Segmenter()
    def forward(self, x):
        """
        Defined the  forward operation.
        """
        x = self.patch_embedding(x)
        x = self.pos_embedding(x)

        x = self.transformer(x)

        # x = segmentation(x)
        return x
    


class ViTConvSegmenter(nn.Module):
    def __init__(self,
                 image_size: int = 256,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 13,
                 embedding_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 hidden_dim: int = 3072,
                 attention_dropout: float = 0.,
                 fc_out_dropout: float = 0.,
                 mlp_dropout: float = 0.
    ):
        
        super(ViTConvSegmenter, self).__init__()

        self.vit = ViT(image_size, patch_size, in_channels, num_classes,embedding_dim, num_layers, num_heads, hidden_dim, attention_dropout, fc_out_dropout, mlp_dropout)
        self.conv_segmenter = ConvSegmenter(1, 13)
        self.patch_merge = PatchMerge(image_size, patch_size, num_classes, 768)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.vit(x)
        
        # masks
        x = self.conv_segmenter(x.unsqueeze(dim=1))
        x = self.patch_merge(x)

        return x



class ViTSegmenter(nn.Module):
    def __init__(self,
                 image_size: int = 256,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 13,
                 embedding_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 hidden_dim: int = 3072,
                 attention_dropout: float = 0.,
                 fc_out_dropout: float = 0.,
                 mlp_dropout: float = 0.,
    ):
        
        super(ViTSegmenter, self).__init__()

        self.vit = ViT(image_size, patch_size, in_channels, num_classes,embedding_dim, num_layers, num_heads, hidden_dim, attention_dropout, fc_out_dropout, mlp_dropout)
        self.out_proj = nn.Parameter(13, 768, 576)

    def forward(self, x):
        x = self.vit(x)
        x = x @ self.out_proj

        return x