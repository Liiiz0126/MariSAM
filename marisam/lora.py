import math
from torch import nn

class LoRALayer(nn.Module):

    def __init__(self, orig_module, r=2, alpha=8):
        super().__init__()
        self.orig_module = orig_module
        self.lora_down = nn.Linear(orig_module.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, orig_module.out_features, bias=False)
        self.scaling = alpha / r
        for p in self.orig_module.parameters():
            p.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.orig_module(x) + self.lora_up(self.lora_down(x)) * self.scaling

    @staticmethod
    def apply_lora_to_encoder(sam_model, r=4, alpha=8):
        """
        递归地遍历 Image Encoder，收集要替换的目标，然后进行替换。
        我们只替换 nn.Linear 层。
        """

        def find_and_replace_linear(module, name_prefix=''):
            for (name, child) in module.named_children():
                new_name = name_prefix + '.' + name if name_prefix else name
                is_linear = isinstance(child, nn.Linear)
                is_qkv_or_proj = any((s in new_name for s in ['attn.qkv', 'attn.proj', 'mlp.layers.1']))
                if is_linear and is_qkv_or_proj:
                    new_lora_layer = LoRALayer(child, r=r, alpha=alpha)
                    setattr(module, name, new_lora_layer)
                else:
                    find_and_replace_linear(child, new_name)
        find_and_replace_linear(sam_model.image_encoder)
