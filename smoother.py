#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.models import vgg19, VGG19_Weights
import torch.utils.checkpoint as checkpoint
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, dataloader
import time

# ulimit -n
# > 1024
# not good for ram. instead use sharing strategy
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings

# from gemini

class VideoDataset(Dataset):
    """Dataset for loading six(?) frames."""
    def __init__(self, files, height=608, width=608, window=5):
        # using ram
        self.height = height
        self.width = width
        self.samples = []
        self.video_cache = [] 
        self.window = window

        print("Pre-loading videos into RAM (This might take a moment)...")

        for vid_idx, vid_path in enumerate(files):
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print(f"Failed to open {vid_path}")
                continue
            
            # 1. Read all frames from this video
            video_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frames.append(frame)
            cap.release()

            if len(video_frames) < window + 1:
                continue

            # Stack into a single tensor for this video: [T, H, W, 3]
            video_tensor = torch.from_numpy(np.stack(video_frames)).share_memory_()
            self.video_cache.append(video_tensor)

            # Create indices
            total_frames = len(video_frames)
            start_t = window // 2 + 1
            end_t = total_frames - window // 2 - 1
            
            # The vid_idx now refers to the index in self.video_cache, 
            # not the original files list (in case some failed to open)
            cache_idx = len(self.video_cache) - 1
            
            if end_t > start_t:
                for t in range(start_t, end_t):
                    self.samples.append((cache_idx, t))

        print(f"Loaded {len(self.video_cache)} videos. Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache_idx, t = self.samples[idx]
        
        # Retrieve the full video tensor from RAM
        video_tensor = self.video_cache[cache_idx] # Shape: [Total_Frames, H, W, 3]
        
        # Slice the 6 frames we need: [t-3 ... t+2]
        # (t-3) is the start index, we need 6 frames total
        padding = self.window // 2 + 1
        frames_uint8 = video_tensor[t-padding : t+padding] 
        
        # Convert to float [0, 1] and permute to [6, 3, H, W]
        # This happens on the fly to save RAM storage
        frames = frames_uint8.permute(0, 3, 1, 2).float() / 255.0
        
        # Window Prev (t-1): Indices 0-4
        window_prev = frames[0:self.window].reshape(-1, self.height, self.width)
        
        # Window Curr (t): Indices 1-5
        window_curr = frames[1:self.window + 1].reshape(-1, self.height, self.width)
        
        return window_curr, window_prev


class ResidualBlock(nn.Module):
    """
    Residual Block: x + Conv(Relu(Conv(x)))
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class DeflickerCNN(nn.Module):
    """
    Multi-frame Convolutional Network for Video Deflickering.
    
    Input: n RGB frames concatenated channel-wise (..., t-2, t-1, t, t+1, t+2, ...).
           Shape: [Batch, input_frames * 3, Height, Width] (Designed for 608x608 inputs)
    Output: 1 RGB frame (stabilized frame t).
            Shape: [Batch, 3, Height, Width]

    Args:
        input_frames: Size of the window. Must be odd.
        num_res_blocks: How many residual blocks to use
        hidden_channels: How many convolution filters per res block
        save_memory: False -> faster compute, but more VRAM; True -> slower compute, less VRAM
            use False for testing, and True for training
    """
    def __init__(self, input_frames=5, num_res_blocks=8, hidden_channels=64, save_memory=False):
        if input_frames % 2 == 0:
            print("Error: input_frames must be odd.")
            exit(-1)
        super(DeflickerCNN, self).__init__()

        self.save_memory = save_memory
        self.input_frames = input_frames
        
        in_channels = input_frames * 3  # frames * 3 channels (R, G, B)       

        # Initial Feature Extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Backbone: Stack of Residual Blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(hidden_channels))
        self.body = nn.Sequential(*res_blocks)
        
        # Final Reconstruction
        self.tail = nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1)

        # force init weights to zero for reconstruction loss during training
        nn.init.constant_(self.tail.weight, 0)
        nn.init.constant_(self.tail.bias, 0)
        
    def forward(self, x):
        features = self.head(x)

        if self.save_memory:
            for layer in self.body:
                features = checkpoint.checkpoint(layer, features, use_reentrant=False)
        else:
            features = self.body(features)
            
        out = self.tail(features)
        
        start_channel = (self.input_frames // 2) * 3 # middle frame * (3 channels R,G,B)
        input_t = x[:, start_channel : start_channel + 3, :, :]
        
        return torch.clamp(out + input_t, 0, 1)


def warp_frame(frame, flow):
    """
    Warps a frame using optical flow.
    
    Args:
        frame: [B, C, H, W] image
        flow: [B, 2, H, W] flow map (2 channels: dx, dy)
    Returns:
        warped_frame: [B, C, H, W]
    """
    B, C, H, W = frame.size()
    
    # Create mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    
    grid = torch.cat((xx, yy), 1).float()
    
    grid = grid.to(frame.device)
    
    vgrid = grid + flow
    
    # Normalize grid to [-1, 1] for grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    # Permute to [B, H, W, 2]
    vgrid = vgrid.permute(0, 2, 3, 1)
    
    # Bilinear sampling
    warped = F.grid_sample(frame, vgrid, align_corners=True, padding_mode='border')
    
    return warped

class RAFT(nn.Module):
    def __init__(self, device):
        super(RAFT, self).__init__()
        self.device = device
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights, progress=False).to(device)
        self.transforms = weights.transforms()

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, img1, img2):
        with torch.no_grad():
            img1_byte = (img1 * 255).byte()
            img2_byte = (img2 * 255).byte()

            img1_pre, img2_pre = self.transforms(img1_byte, img2_byte)

            # 3. Run the model and get the flow
            flow = self.model(img1_pre, img2_pre)[-1]
            
            # 4. Return the flow to the original device (so it matches output_prev later)
            return flow
            

class DeflickerLoss(nn.Module):
    """
    Combined loss: Temporal Consistency + Reconstruction
    """
    def __init__(self, lambda_l1=1.0, lambda_rec=1.0, lambda_perc=1.0, device=settings.VGG_DEVICE_ID if settings.USE_VGG_DEVICE else "cuda"):
        super(DeflickerLoss, self).__init__()
        self.lambda_rec = lambda_rec
        self.lambda_perc = lambda_perc
        self.lambda_l1 = lambda_l1
        self.l1_loss = nn.L1Loss()
        vgg = vgg19(weights=VGG19_Weights).features
        self.vgg = vgg[:29].to(device).eval()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # magic numbers from Gemini
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))


    def forward(self, output_t, input_t, output_prev, flow, occlusion_mask=None):
        """
        Args:
            output_t: Network output for current frame t
            input_t: Original input frame t (flickering source)
            output_prev: Network output for previous frame t-1
            flow_prev_to_curr: Optical flow from t-1 to t
            occlusion_mask: (Optional) Weight mask where 0 indicates occlusion/new content
                            and 1 indicates valid tracking.
        """
        # 1. Reconstruction Loss
        if input_t.device != output_t.device:
            input_t = input_t.to(output_t.device)
        rec_loss = self.l1_loss(output_t, input_t)

        # 2: VGG Loss
        vgg_device = self.mean.device
            
        # Normalize inputs: (x - mean) / std
        # We must move the inputs to vgg_device to perform the forward pass
        out_norm = (output_t.to(vgg_device) - self.mean) / self.std
        in_norm = (input_t.to(vgg_device) - self.mean) / self.std

        # Extract features
        out_feat = self.vgg(out_norm)
        in_feat = self.vgg(in_norm)

        # Calculate feature loss
        perc_loss = self.l1_loss(out_feat, in_feat)
         
        # 3. Temporal Loss (Warping Loss)
        # Hopefully to eliminate device shenanigains 
        if occlusion_mask is not None and occlusion_mask.device != output_t.device:
            occlusion_mask = occlusion_mask.to(output_t.device)
        # warp_frame will now work because we fixed it to use input.device
        warped_prev = warp_frame(output_prev, flow)
        
        if occlusion_mask is not None:
            temp_diff = torch.abs(output_t - warped_prev) * occlusion_mask
            temp_loss = torch.mean(temp_diff)
        else:
            temp_loss = self.l1_loss(output_t, warped_prev)
            
        total_loss = (self.lambda_l1 * temp_loss) + (self.lambda_rec * rec_loss) + self.lambda_perc * perc_loss.to(output_t.device)
        
        return total_loss, temp_loss, rec_loss, perc_loss


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Takes in (model, optimizer, checkpoint_path, device) and returns start_epochs.

    Note: model and optimizer are passed by reference or similar, so in C it would be &model, &optimizer
    """
    checkpoint = torch.load(checkpoint_path, device)
    model_state = checkpoint['model_state_dict']
    optimizer_state = checkpoint['optimizer_state_dict']

    new_state_dict = {}
    for k, v in model_state.items():
        if k.startswith('module.'):
            # remove the first 7 characters ('module.')
            new_key = k[7:]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict) 
    optimizer.load_state_dict(optimizer_state)


    return checkpoint.get('epoch', 0) + 1


def generate_circle_mask(height=608, width=608, radius=250, vertical_offset=15, device="cuda" if torch.cuda.is_available() else "cpu"):
    center_x = width // 2
    # In the PIL code provided, top = center_y - radius - vertical_offset. 
    # The mathematical center of that circle is (center_x, center_y - vertical_offset).
    center_y = (height // 2) - vertical_offset
    
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    
    dist_sq = (x - center_x)**2 + (y - center_y)**2
    mask = (dist_sq <= radius**2).float()
    
    return mask.view(1, 1, height, width)


    
def main(
    data_dir=os.path.join("data", "images"),
    input_frames=5,
    num_res_blocks=24,
    hidden_channels=256,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    LAMBDA=0.5,
    previous_model_path=None,
    debug=False
):

    if not os.path.exists(settings.MODEL_SAVE_PATH):
        os.mkdir(settings.MODEL_SAVE_PATH)

    start_epoch = 0

    if settings.USE_DEVICE_IDS:
        device = torch.device(f"cuda:{settings.DEVICE_IDS[0]}")
    

    print("Initializing model")
    model = DeflickerCNN(input_frames=input_frames, num_res_blocks=num_res_blocks, hidden_channels=hidden_channels, save_memory=True,)
    model.to(device)

    criterion = DeflickerLoss(
        lambda_l1=0.0, lambda_perc=1.0, lambda_rec=LAMBDA, device = settings.VGG_DEVICE_ID if settings.USE_VGG_DEVICE else device
    ).to(settings.VGG_DEVICE_ID if settings.USE_VGG_DEVICE else device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    if previous_model_path and os.path.exists(previous_model_path):
        print(f"Peeking at {previous_model_path} for architecture...")
        temp_ckpt = torch.load(previous_model_path, map_location='cpu')
        
        if 'num_res_blocks' in temp_ckpt:
            num_res_blocks = temp_ckpt['num_res_blocks']
        if 'hidden_channels' in temp_ckpt:
            hidden_channels = temp_ckpt['hidden_channels']
        if 'input_frames' in temp_ckpt:
            input_frames = temp_ckpt['input_frames']
        
        del temp_ckpt
        print(f"Resuming with: Depth={num_res_blocks}, Width={hidden_channels}")

        print(f"Loading previous model from {previous_model_path}")
        start_epoch = load_checkpoint(model, optimizer, previous_model_path, device)

    if settings.USE_DEVICE_IDS:
        print("Wrapping model with nn.DataParallel")
        model = nn.DataParallel(model, device_ids=settings.DEVICE_IDS)

    raft_model = RAFT(settings.VGG_DEVICE_ID if settings.USE_VGG_DEVICE else device)
    if settings.USE_DEVICE_IDS:
        raft_model = nn.DataParallel(raft_model, device_ids=settings.DEVICE_IDS)

    print("Initializing datasets")
    data_arr = [os.path.join(data_dir, video) for video in os.listdir(data_dir)]
    if settings.NUM_IMAGES > 0:
        data_arr = data_arr[: settings.NUM_IMAGES] # reusing NUM_IMAGES to be number of videos
    split_idx = int(len(data_arr) * settings.VALUE_SPLIT)

    train_files = data_arr[: split_idx]
    valid_files = data_arr[split_idx :]

    train_dataset = VideoDataset(train_files, window=input_frames)
    valid_dataset = VideoDataset(valid_files, window=input_frames)

    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=settings.NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=settings.BATCH_SIZE, shuffle=False, num_workers=settings.NUM_WORKERS)

    if not len(train_loader) > 0 or not len(valid_loader) > 0:
        print("error: no training and/or validation samples found. womp womp.")
        exit()

    print("Generating mask")
    roi_mask = generate_circle_mask(height=train_dataset.height, width=train_dataset.width, device=settings.VGG_DEVICE_ID if settings.USE_VGG_DEVICE else device)

    for epoch in range(start_epoch, settings.NUM_EPOCHS):
        start_time = time.time()
        print(f"Epoch {epoch+1} / {settings.NUM_EPOCHS}")
        running_loss = 0.0

        if debug:
            print("Debug ending. Exiting . . .")
            exit()

        # train
        print("  Beginning training")
        model.train()
        for batch_idx, (inputs_curr, inputs_prev) in enumerate(train_loader):
            inputs_curr = inputs_curr.to(device)
            inputs_prev = inputs_prev.to(device)

            optimizer.zero_grad()

            input_frame_t = inputs_curr[:, 6:9, :, :]

            input_frame_prev = inputs_curr[:, 3:6, :, :] 
            input_frame_curr = inputs_curr[:, 6:9, :, :]

            flow = raft_model(input_frame_prev, input_frame_curr)
            with torch.autocast(device_type='cuda'):
                output_t = model(inputs_curr)
                with torch.no_grad():
                    output_prev = model(inputs_prev)

                total_loss, t_loss, r_loss, p_loss = criterion(
                    output_t=output_t,
                    input_t=input_frame_t,
                    output_prev=output_prev,
                    flow=flow,
                    occlusion_mask=roi_mask
                )

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if batch_idx % 20 == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)} | Total Loss: {total_loss.item():.4f} | Temp: {t_loss.item():.4f} |  Rec: {r_loss.item():.4f} | Perc: {p_loss.item():.4f} | Time: {time.time() - start_time:.2f}s")

        print(f"  Training finished in {(time.time() - start_time):4f}s | Total Loss: {running_loss/len(train_loader):.4f}") 

        # validation
        print("  Beginning validation")
        validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs_curr, inputs_prev) in enumerate(valid_loader):
                inputs_curr = inputs_curr.to(device)
                inputs_prev = inputs_prev.to(device)

                input_frame_t = inputs_curr[:, 6:9, :, :]
                input_frame_prev = inputs_curr[:, 3:6, :, :] 
                input_frame_curr = inputs_curr[:, 6:9, :, :]

                flow = raft_model(input_frame_prev, input_frame_curr)
                with torch.autocast(device_type='cuda'):
                    output_t = model(inputs_curr)
                    output_prev = model(inputs_prev)

                    total_loss, t_loss, r_loss, p_loss = criterion(
                        output_t=output_t,
                        input_t=input_frame_t,
                        output_prev=output_prev,
                        flow=flow,
                        occlusion_mask=roi_mask
                    )

                validation_loss += total_loss.item()

                if batch_idx % 20 == 0:
                    print(f"    Batch {batch_idx}/{len(valid_loader)} | Total Loss: {total_loss.item():.4f} | Temp: {t_loss.item():.4f} |  Rec: {r_loss.item():.4f} | Perc: {p_loss.item():.4f} | Time: {time.time() - start_time:.2f}s")

        print(f"  Training finished in {(time.time() - start_time):4f}s | Total Loss: {validation_loss/len(valid_loader):.4f}") 


        avg_loss = running_loss / len(train_loader)
        avg_val = validation_loss / len(valid_loader)
        print(f"Epoch {epoch+1} Complete. \nAverage Loss: {avg_loss:.4f} | Average Validation Loss {avg_val:.4f}")
        print(f"Epoch duration: {time.time() - start_time:.2f}s")
 
        if epoch % settings.EPOCH_SAVE_INTERVAL == 0:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_frames': input_frames,
                'num_res_blocks': num_res_blocks,
                'hidden_channels': hidden_channels,
            }
            model_name = f"checkpoint_epoch_{epoch}.pth"
            torch.save(
                save_dict,
                os.path.join(settings.MODEL_SAVE_PATH, model_name),
            )
            print(f"Model saved to {settings.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main(
        data_dir=os.path.join('media', 'filtered_training_videos'),
        debug=True
    )
