import torch
from torch import nn

class LayerNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class NIRGatingUnit(nn.Module):

    def __init__(self, mid_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(nn.Conv2d(1, mid_channels // 2, kernel_size=1), nn.ReLU(), nn.Conv2d(mid_channels // 2, mid_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, rgb_feat, nir_map):
        gate = self.gate_conv(nir_map)
        return rgb_feat + rgb_feat * gate

class IndexContextModulator(nn.Module):

    def __init__(self, num_indices, feat_channels):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(num_indices, feat_channels, kernel_size=1), nn.ReLU())
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(feat_channels, feat_channels // 2), nn.ReLU(), nn.Linear(feat_channels // 2, feat_channels), nn.Sigmoid())

    def forward(self, main_feat, indices_map):
        """
        main_feat:   [B, C, H, W]
        indices_map: [B, N, H, W]
        """
        idx_feat = self.project(indices_map)
        idx_vec = self.global_pool(idx_feat).flatten(1)
        channel_weights = self.mlp(idx_vec).view(-1, idx_feat.shape[1], 1, 1)
        return main_feat + main_feat * channel_weights

class UniversalSpectralStyleAdapter(nn.Module):

    def __init__(self, in_bands=4, out_channels=3, mid_channels=32, use_indices=False, indices_list=None):
        """
        indices_list (list of str): 例如 ['ndvi', 'ndwi', 'r1']. 
                                    如果为 None 且 use_indices=True，则使用默认组合.
        """
        super().__init__()
        self.use_indices = use_indices
        self.mid_channels = mid_channels
        if self.use_indices:
            if indices_list is None or len(indices_list) == 0:
                self.indices_list = ['ndvi', 'ndwi']
            else:
                self.indices_list = [x.lower() for x in indices_list]
            self.num_indices = len(self.indices_list)
        else:
            self.indices_list = []
            self.num_indices = 0
        self.rgb_entry = nn.Sequential(nn.Conv2d(3, mid_channels, kernel_size=1, bias=False), LayerNorm2d(mid_channels), nn.GELU())
        self.nir_gating = NIRGatingUnit(mid_channels)
        if self.use_indices and self.num_indices > 0:
            self.index_modulator = IndexContextModulator(self.num_indices, mid_channels)
        self.branch_small = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False), LayerNorm2d(mid_channels), nn.GELU())
        self.branch_large = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False), LayerNorm2d(mid_channels), nn.GELU())
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.style_fc = nn.Sequential(nn.Linear(mid_channels, mid_channels // 2), nn.ReLU(), nn.Linear(mid_channels // 2, mid_channels * 2), nn.Sigmoid())
        self.exit_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        nn.init.constant_(self.exit_conv.weight, 0)
        nn.init.constant_(self.exit_conv.bias, 0)

    def compute_indices(self, x):
        """
        动态根据 self.indices_list 计算并返回指数
        """
        eps = 1e-08
        b = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        r = x[:, 2:3, :, :]
        n = x[:, 3:4, :, :]
        ndvi = (n - r) / (n + r + eps)
        ndwi = (g - n) / (g + n + eps)
        rvi = n / (r + eps)
        bg_avg = (b + g) / 2
        ndai = (bg_avg - n) / (bg_avg + n + eps)
        mai = (g + b) / 2 - r
        ndawi = (b + g - (r + n)) / (b + g + (r + n) + eps)
        r1 = g / (n + eps)
        r2 = g / (r + eps)
        r3 = b / (r + eps)
        index_dict = {'ndvi': ndvi, 'ndwi': ndwi, 'rvi': rvi, 'ndai': ndai, 'mai': mai, 'ndawi': ndawi, 'r1': r1, 'r2': r2, 'r3': r3}
        selected_tensors = []
        for name in self.indices_list:
            if name in index_dict:
                selected_tensors.append(index_dict[name])
            else:
                raise KeyError(f"Index '{name}' not found in supported list: {list(index_dict.keys())}")
        return torch.cat(selected_tensors, dim=1)

    def forward(self, x):
        (B, C, H, W) = x.shape
        if C == 4:
            rgb_input = x[:, 0:3, :, :].flip(1)
            nir_input = x[:, 3:4, :, :]
            has_nir = True
        elif C == 3:
            rgb_input = x
            nir_input = None
            has_nir = False
        else:
            raise ValueError(f'Input channels must be 3 or 4, got {C}')
        rgb_feat = self.rgb_entry(rgb_input)
        if has_nir:
            feat = self.nir_gating(rgb_feat, nir_input)
        else:
            feat = rgb_feat
        if has_nir and self.use_indices and (self.num_indices > 0):
            indices_map = self.compute_indices(x)
            feat = self.index_modulator(feat, indices_map)
        style_vec = self.global_pool(feat).flatten(1)
        style_weights = self.style_fc(style_vec).view(B, 2, -1, 1, 1)
        feat_small = self.branch_small(feat)
        feat_large = self.branch_large(feat)
        fused_feat = feat_small * style_weights[:, 0] + feat_large * style_weights[:, 1]
        delta = self.exit_conv(fused_feat)
        out = rgb_input + delta
        return out
