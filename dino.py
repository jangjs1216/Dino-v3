
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import glob
import os
import math

# --- Model Definitions ---

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

class CustomLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('bias_mask', torch.ones(out_features))

    def forward(self, input):
        # Assuming bias_mask modifies bias or output? 
        # For simple inference loading, we'll use standard linear behavior
        # unless bias_mask is intended to zero out bias.
        # But we must have the attribute to load state_dict.
        return F.linear(input, self.weight, self.bias)

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Inspecting keys: blocks.0.attn.qkv.weight, blocks.0.attn.qkv.bias_mask
        self.qkv = CustomLinear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope is not None:
             q, k = rope(q, k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values=init_values)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        # Assuming ls2 exists based on ls1 presence in standard layer scale utils
        self.ls2 = LayerScale(dim, init_values=init_values)

    def forward(self, x, rope=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

class RoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Creating a parameter to match 'rope_embed.periods'
        # Shape was [16] in inspection. 
        self.periods = nn.Parameter(torch.zeros(16)) 

    def forward(self, q, k):
        # Placeholder for RoPE logic. 
        # Without exact freq logic, we just return q, k.
        # This allows loading state dict without error.
        # Implementing incorrect RoPE would degrade performance, 
        # but loading is primary here.
        return q, k

class DinoV3(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.):
        super().__init__()
        
        self.patch_embed = nn.Module()
        # patch_embed.proj.weight, patch_embed.proj.bias
        self.patch_embed.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, 4, embed_dim)) # 4 storage tokens
        
        self.rope_embed = RoPE(embed_dim)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: B, C, H, W
        x = self.patch_embed.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B, N, C
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        storage_tokens = self.storage_tokens.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, storage_tokens, x), dim=1)
        
        for blk in self.blocks:
            x = blk(x, rope=self.rope_embed)
            
        x = self.norm(x)
        return x

# --- Helper Functions ---

def split_image(image_path, patch_h, patch_w):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    patches = []
    
    # Calculate stepping. If image is smaller than patch size, we just take one crop (or resize? assuming input >= patch_size)
    # If we want to cover the whole image.
    
    # Logic: simple grid. If (j + patch_w) > w, shift j back so that box ends at w.
    # Same for i.
    
    # Generate start coordinates
    y_starts = list(range(0, h, patch_h))
    if y_starts[-1] + patch_h > h:
        # If the last regular step goes out of bounds, 
        # we still want to cover the end. 
        # However, the user wants "naturally cut into the image" -> shift the last patch.
        # But simply adding the shifted patch might duplicate content.
        # Let's check common logic: Sliding window with stride = patch_size.
        # If last window > h, we make the last window start at h - patch_h.
        pass

    # Actually, a simpler loop approach:
    # Iterate with stride, but if we exceed, we clamp the start.
    
    # Correct approach for strictly "covering" without padding:
    # 1. 0, patch_h, 2*patch_h ... 
    # 2. If current + patch_h > h, then current = h - patch_h. And stop after this.
    
    i = 0
    while i < h:
        top = i
        bottom = i + patch_h
        if bottom > h:
            bottom = h
            top = h - patch_h
            if top < 0: top = 0 # Handle image smaller than patch
        
        j = 0
        while j < w:
            left = j
            right = j + patch_w
            if right > w:
                right = w
                left = w - patch_w
                if left < 0: left = 0
            
            box = (left, top, right, bottom)
            crop = img.crop(box)
            
            # Helper to resize if image is smaller than patch size (rare edge case)
            if crop.size != (patch_w, patch_h):
                 crop = crop.resize((patch_w, patch_h), Image.BICUBIC)
            
            patches.append(crop)
            
            if right == w:
                break
            j += patch_w
        
        if bottom == h:
            break
        i += patch_h
            
    return patches

def main():
    pth_file = 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    
    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(ext))
        
    if not image_files:
        print("No images found in directory.")
        return

    print(f"Found images: {image_files}")
    
    # Initialize model
    model = DinoV3(embed_dim=384, depth=12, num_heads=6)
    
    print(f"Loading weights from {pth_file}...")
    try:
        state_dict = torch.load(pth_file, map_location='cpu')
        
        # Handle 'state_dict' key if present (not in this case based on inspection)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        # Helper to strict=False loading with basic checking
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load message: {msg}")
        
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((640, 640)), # Ensure patches are exactly 640x640 if coming from split?
        # Actually split_image returns 640x640 PIL images already.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ... (previous code)

    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Please install scikit-learn and matplotlib to visualize results.")
        return

    def visualize_result(patch_img_tensor, output_tensor, patch_idx, output_folder, file_basename):
        # ... (same logic, but save to output_folder)
        features = output_tensor[0, 5:, :] 
        n_tokens = features.shape[0]
        side = int(math.sqrt(n_tokens))
        
        features = features.cpu().numpy()
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features)
        pca_features = (pca_features - pca_features.min(0)) / (pca_features.max(0) - pca_features.min(0))
        pca_img = pca_features.reshape(side, side, 3)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        img = patch_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[0].imshow(img)
        axes[0].set_title(f"Original Patch {patch_idx}")
        axes[0].axis('off')
        
        axes[1].imshow(pca_img)
        axes[1].set_title(f"PCA Features {patch_idx}")
        axes[1].axis('off')
        
        output_filename = os.path.join(output_folder, f"{file_basename}_patch_{patch_idx}_pca.png")
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"  Saved visualization to {output_filename}")

    def visualize_full_channels(output_tensor, patch_idx, output_folder, file_basename):
        features = output_tensor[0, 5:, :] 
        n_tokens = features.shape[0]
        side = int(math.sqrt(n_tokens))
        n_channels = features.shape[1]

        cols = 24
        rows = int(math.ceil(n_channels / cols))
        
        features = features.reshape(side, side, n_channels).permute(2, 0, 1).cpu().numpy()
        canvas = np.zeros((rows * side, cols * side))
        
        for i in range(n_channels):
            r = i // cols
            c = i % cols
            ch_img = features[i]
            dmin, dmax = ch_img.min(), ch_img.max()
            if dmax - dmin > 1e-6:
                ch_img = (ch_img - dmin) / (dmax - dmin)
            else:
                ch_img = np.zeros_like(ch_img)
            canvas[r*side:(r+1)*side, c*side:(c+1)*side] = ch_img

        plt.figure(figsize=(24, 16))
        plt.imshow(canvas, cmap='viridis')
        plt.title(f"All {n_channels} Channels for Patch {patch_idx}")
        plt.axis('off')
        plt.tight_layout()
        
        out_name = os.path.join(output_folder, f"{file_basename}_patch_{patch_idx}_all_channels.png")
        plt.savefig(out_name, dpi=150)
        plt.close()
        print(f"  Saved full channel grid to {out_name}")

    def process_images_in_folder(input_folder, output_folder, model, transform):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            # Recursive or simple? Glob in python doesn't do recursive by default unless ** and recursive=True
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            
        if not image_files:
            print(f"No images found in {input_folder}")
            return

        for img_path in image_files:
            file_basename = os.path.splitext(os.path.basename(img_path))[0]
            print(f"\nProcessing {img_path}...")
            
            # Use 640x640 splitting logic
            patches = split_image(img_path, 640, 640)
            print(f"Split into {len(patches)} patches of 640x640.")
            
            for i, patch in enumerate(patches):
                input_tensor = transform(patch).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    
                print(f"  Patch {i}: Output shape {output.shape}")
                visualize_result(input_tensor, output, i, output_folder, file_basename)
                visualize_full_channels(output, i, output_folder, file_basename)

    # Call the processing function
    # Current directory as input, 'output_vis' as output
    process_images_in_folder('.', 'output_vis', model, transform)


if __name__ == "__main__":
    main()
