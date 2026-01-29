
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
        return F.linear(input, self.weight, self.bias)

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
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
        self.ls2 = LayerScale(dim, init_values=init_values)

    def forward(self, x, rope=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

class RoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.periods = nn.Parameter(torch.zeros(16)) 

    def forward(self, q, k):
        return q, k

class DinoV3(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, 4, embed_dim))
        
        self.rope_embed = RoPE(embed_dim)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        storage_tokens = self.storage_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, storage_tokens, x), dim=1)
        
        for blk in self.blocks:
            x = blk(x, rope=self.rope_embed)
            
        x = self.norm(x)
        return x

# --- Feature Database & Search Logic ---

class FeatureDatabase:
    def __init__(self):
        self.features = [] # List of (N, 384) tensors
        self.image_paths = []
        self.patch_counts = [] # Number of patches (N) per image
        self.metadata = [] # List of dicts with 'grid_size': (rows, cols)

    def add_image(self, img_path, features, grid_size):
        # features: (1, N_tokens, 384) - we want to store relevant tokens
        # Remove CLS (0) and Storage (1-4). Keep 5:
        # features shape after slice: (N_patches, 384)
        patch_features = features[0, 5:, :].cpu()
        
        self.features.append(patch_features)
        self.image_paths.append(img_path)
        self.patch_counts.append(patch_features.shape[0])
        self.metadata.append({'grid_size': grid_size})

    def build_flattened_index(self):
        # Concatenate all features for global PCA
        if not self.features:
            print("Database is empty.")
            return None
        
        # Keep track of which global index belongs to which image
        self.global_index_map = []
        start_idx = 0
        for i, count in enumerate(self.patch_counts):
            # Range [start_idx, start_idx + count) belongs to image i
            self.global_index_map.append((start_idx, count))
            start_idx += count
            
        all_features = torch.cat(self.features, dim=0) # (Total_Patches, 384)
        
        # --- Perform PCA to reduce to 3 dimensions ---
        print(f"Applying PCA (384 -> 3) on {all_features.shape[0]} patches...")
        
        # Center the data
        mean = all_features.mean(dim=0)
        centered_features = all_features - mean
        
        # Compute PCA using SVD
        # We want the top 3 principal components
        try:
            # torch.pca_lowrank is efficient
            U, S, V = torch.pca_lowrank(centered_features, q=3, center=False, niter=2)
            # Project data: (N, 384) @ (384, 3) -> (N, 3)
            self.flattened_features = torch.matmul(centered_features, V[:, :3])
        except Exception as e:
            print(f"PCA failed: {e}. Fallback to random projection.")
            self.flattened_features = torch.randn(all_features.shape[0], 3)
            
        print(f"Database built: {len(self.image_paths)} images, {self.flattened_features.shape[0]} total patches (3D).")

        # Update self.features list to store the 3D versions for easy access per image
        new_features_list = []
        idx = 0
        for count in self.patch_counts:
            new_features_list.append(self.flattened_features[idx : idx+count])
            idx += count
        self.features = new_features_list

    def search(self, query_vector, exclude_global_idx=None, exclude_img_idx=None, top_k=5):
        # query_vector: (3,)
        if hasattr(self, 'flattened_features') is False:
            self.build_flattened_index()
            
        if query_vector.dim() == 1:
            query_vector = query_vector.unsqueeze(0) # (1, 3)
            
        # Euclidean distance
        # dists: (Total,)
        dists = torch.cdist(query_vector, self.flattened_features, p=2).squeeze(0)
        
        # We want the smallest distance
        n_candidates = min(top_k * 10, dists.shape[0])
        top_dists, top_indices = torch.topk(dists, n_candidates, largest=False)
        
        results = []
        found_count = 0
        
        for dist, global_idx in zip(top_dists, top_indices):
            if found_count >= top_k:
                break
                
            global_idx = global_idx.item()
            dist = dist.item()

            # Filter: Skip specific excluded index (self-match)
            if exclude_global_idx is not None and global_idx == exclude_global_idx:
                continue
            
            # Find which image this belongs to
            found_img_idx = -1
            local_idx = -1
            
            # Binary search or simple linear scan
            for img_idx, (start, count) in enumerate(self.global_index_map):
                if start <= global_idx < start + count:
                    found_img_idx = img_idx
                    local_idx = global_idx - start
                    break
            
            if found_img_idx != -1:
                # Exclude specific image index (e.g. current query image)
                if exclude_img_idx is not None and found_img_idx == exclude_img_idx:
                    continue
                    
                img_path = self.image_paths[found_img_idx]

                # Calculate row/col
                grid_rows = self.metadata[found_img_idx]['grid_size'][0] 
                grid_cols = self.metadata[found_img_idx]['grid_size'][1] 
                
                row = local_idx // grid_cols
                col = local_idx % grid_cols
                
                patch_box = self.metadata[found_img_idx].get('patch_box')

                # One match per image file constraint
                # If we already have a result from this image (img_path), skip this candidate
                if any(res['image_path'] == img_path for res in results):
                    continue
                
                results.append({
                    'image_path': img_path,
                    'score': dist,
                    'patch_row': row,
                    'patch_col': col,
                    'grid_size': (grid_rows, grid_cols),
                    'patch_box': patch_box
                })
                found_count += 1
                
        return results

# --- Helper Functions ---

def split_image_with_coords(img, patch_h, patch_w):
    w, h = img.size
    patches = []
    coords = [] # (left, top, right, bottom)
    
    i = 0
    while i < h:
        top = i
        bottom = i + patch_h
        if bottom > h:
            bottom = h
            top = h - patch_h
            if top < 0: top = 0
        
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
            if crop.size != (patch_w, patch_h):
                 crop = crop.resize((patch_w, patch_h), Image.BICUBIC)
            
            patches.append(crop)
            coords.append(box)
            
            if right == w: break
            j += patch_w
        if bottom == h: break
        i += patch_h
        
    return patches, coords

def interactive_search_session(db, model, transform, top_k=5):
    import matplotlib
    try:
        matplotlib.use('Qt5Agg')
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    if not db.image_paths:
        print("No images in database.")
        return

    # Helper: Load image and setup plot
    current_img_idx = 0
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.2)
    
    img_data = {'patches': [], 'coords': [], 'features': None}
    
    def load_current_image():
        ax.clear()
        img_path = db.image_paths[current_img_idx]
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        ax.set_title(f"Click on a defect! [{current_img_idx+1}/{len(db.image_paths)}] {os.path.basename(img_path)}")
        ax.axis('off')
        
        # We need patch coords and features.
        # DB features are stored by linear index in self.flattened, or by lists in self.features.
        # self.features[idx] corresponds to the idx-th entry in DB.
        # In process_and_build_db, each entry is ONE 640x640 patch.
        # But here we are loading the FULL image file.
        # If the image was split into multiple patches in DB, we have multiple DB entries for one file.
        # This complicates "load_current_image" if we want to show the full image and let user click anywhere.
        
        # Simplified approach: Treat each DB entry (640x640 patch) as a separate "image" to browse.
        # So "current_img_idx" iterates over DB entries (patches).
        # We visualize the crop.
        
        # If the user wants to see the full image and click, we'd need to stitch.
        # Given "640x640 이미지에서 유저가 특정 불량 지점을 선택하면", let's stick to showing the 640x640 patch.
        
        patch_box = db.metadata[current_img_idx].get('patch_box')
        # Load full image then crop to patch_box
        full_img = Image.open(img_path).convert('RGB')
        # crop
        patch_img = full_img.crop(patch_box)
        
        ax.clear()
        ax.imshow(patch_img)
        ax.set_title(f"Entry {current_img_idx}: {os.path.basename(img_path)}")
        ax.axis('off')

        img_data['features'] = db.features[current_img_idx] # (1600, 384)
        plt.draw()

    load_current_image()
    
    # Navigation Buttons
    from matplotlib.widgets import Button
    axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Prev')

    def next_img(event):
        nonlocal current_img_idx
        current_img_idx = (current_img_idx + 1) % len(db.image_paths)
        load_current_image()

    def prev_img(event):
        nonlocal current_img_idx
        current_img_idx = (current_img_idx - 1) % len(db.image_paths)
        load_current_image()

    bnext.on_clicked(next_img)
    bprev.on_clicked(prev_img)

    def on_click(event):
        if event.inaxes != ax: return
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked at: {x}, {y}")
        
        # Visual Feedback
        [p.remove() for p in ax.patches if not hasattr(p, 'is_result_box')]
        ax.plot(x, y, 'rx', markersize=10)
        
        # Map pixel to feature index (40x40 grid)
        # 640 / 40 = 16 pixels per block
        grid_row = int(y / 16)
        grid_col = int(x / 16)
        
        if grid_row >= 40: grid_row = 39
        if grid_col >= 40: grid_col = 39
        
        feature_idx = grid_row * 40 + grid_col
        
        target_vector = None
        if feature_idx < img_data['features'].shape[0]:
            target_vector = img_data['features'][feature_idx]
            print(f"Selecting Feature Grid ({grid_row}, {grid_col})")
            
        if target_vector is not None:
            # Calculate global index of the query to exclude it precisely
            # db.global_index_map[img_idx] -> (start, count)
            start_global_idx, _ = db.global_index_map[current_img_idx]
            query_global_idx = start_global_idx + feature_idx
            
            results = db.search(target_vector, exclude_global_idx=query_global_idx, exclude_img_idx=current_img_idx, top_k=top_k)
            show_results(results, (x, y))

    def show_results(results, query_xy):
        if not results:
            print("No matches found (other than source).")
            return
            
        n_results = len(results)
        res_fig, res_axes = plt.subplots(1, n_results, figsize=(3 * n_results, 5))
        
        # Handle singleton axis if n_results = 1
        if n_results == 1:
            res_axes = [res_axes]
            
        for k, res in enumerate(results):
            ax_k = res_axes[k]
            
            r_path = res['image_path']
            r_score = res['score']
            p_box = res['patch_box'] # (left, top, right, bottom) in original image
            
            full_img = Image.open(r_path).convert('RGB')
            patch_img = full_img.crop(p_box)
            
            ax_k.imshow(patch_img)
            ax_k.set_title(f"{r_score:.3f}\n{os.path.basename(r_path)}")
            ax_k.axis('off')
            
            # Draw Red Box around the specific 16x16 feature block
            r_row = res['patch_row']
            r_col = res['patch_col']
            
            # 1 feature = 16x16 pixels
            rect_x = r_col * 16
            rect_y = r_row * 16
            
            rect = Rectangle((rect_x, rect_y), 16, 16, linewidth=2, edgecolor='red', facecolor='none')
            ax_k.add_patch(rect)
            
        res_fig.show()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

def process_and_build_db(input_folder, model, transform):
    db = FeatureDatabase()
    image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + \
                  glob.glob(os.path.join(input_folder, "*.png"))
    
    print(f"Building database from {len(image_files)} image files...")
    
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        patches, coords = split_image_with_coords(img, 640, 640)
        
        for i, (patch, box) in enumerate(zip(patches, coords)):
            input_tensor = transform(patch).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor) # (1, 1605, 384)
            
            db.add_image(
                img_path, 
                output, 
                grid_size=(40, 40)
            )
            db.metadata[-1]['patch_box'] = box
            db.metadata[-1]['patch_index'] = i

    db.build_flattened_index()
    return db

def main():
    pth_file = 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    if not os.path.exists(pth_file):
        print(f"Model file {pth_file} not found.")
        return

    model = DinoV3(embed_dim=384, depth=12, num_heads=6)
    try:
        state_dict = torch.load(pth_file, map_location='cpu')
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((640, 640)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Building Database...")
    db = process_and_build_db('.', model, transform)
        
    print("Starting interactive session...")
    print("A window will open. Click on an image to find similar patches.")
    interactive_search_session(db, model, transform, top_k=5)

if __name__ == "__main__":
    main()
