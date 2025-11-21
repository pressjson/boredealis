#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time

if not os.path.exists("local_settings.py"):
    print("Warning: local settings not found. Using default settings.")
    import settings
else:
    import local_settings as settings

# from gemini

class VideoDataset(Dataset):
    """Dataset for loading six(?) frames."""
    def __init__(self, root, height=608, width=608):
        self.root_dir = root
        self.height = height
        self.width = width
        self.video_files = [os.path.join(self.root_dir, video) for video in os.listdir(self.root_dir)]

        self.samples = []

        print("Indexing frames . . .")

        for vid_idx, vid_path in enumerate(self.video_files):
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # We need a continuous block of 6 frames: [t-3, t-2, t-1, t, t+1, t+2]
            # So t must be >= 3 and t+2 < total_frames
            # t >= 3
            # t <= total_frames - 3
            start_t = 3
            end_t = total_frames - 3
            
            if end_t > start_t:
                for t in range(start_t, end_t):
                    self.samples.append((vid_idx, t))
            
            cap.release()

        print(f"Indexed {len(self.samples)} samples. Sheeeeeeeesh.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_idx, t = self.samples[idx]
        vid_path = self.video_files[vid_idx]
        
        # Open video
        cap = cv2.VideoCapture(vid_path)

        start_frame_idx = t - 3
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        
        frames = []
        for _ in range(6):
            ret, frame = cap.read()
            if not ret:
                # Fallback: pad with zeros if read fails (shouldn't happen if index correct)
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
            
        cap.release()
        
        # Stack frames: [6, H, W, 3] -> [6, 3, H, W]
        frames_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        
        # Window Prev (t-1): Indices 0,1,2,3,4 -> correspond to t-3...t+1
        window_prev = frames_tensor[0:5].reshape(-1, self.height, self.width) # flatten channels: 5*3=15
        
        # Window Curr (t): Indices 1,2,3,4,5 -> correspond to t-2...t+2
        window_curr = frames_tensor[1:6].reshape(-1, self.height, self.width)
        
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
    
    Input: 5 RGB frames concatenated channel-wise (t-2, t-1, t, t+1, t+2).
           Shape: [Batch, 15, Height, Width] (Designed for 608x608 inputs)
    Output: 1 RGB frame (stabilized frame t).
            Shape: [Batch, 3, Height, Width]
    """
    def __init__(self, input_frames=5, num_res_blocks=8, hidden_channels=64):
        super(DeflickerCNN, self).__init__()
        
        in_channels = input_frames * 3  # 5 frames * 3 channels = 15
        
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
        
    def forward(self, x):
        # x shape: [B, 15, H, W]
        
        features = self.head(x)
        features = self.body(features)
        out = self.tail(features)
        
        # Global Residual connection: Input t (center frame) + Network Output
        # The network learns the correction mask to remove flicker
        center_frame_index = 2 * 3 # Start index of frame t (frames are 0,1,2,3,4)
        input_t = x[:, center_frame_index:center_frame_index+3, :, :]
        
        # Clamp output to valid image range
        return torch.sigmoid(out + input_t)



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
        # 1. Capture the device of the inputs (where your main loop is running)
        original_device = img1.device
        
        # 2. Move inputs to the RAFT model's device (if different)
        if img1.device != self.device:
            img1 = img1.to(self.device)
        if img2.device != self.device:
            img2 = img2.to(self.device)

        with torch.no_grad():
            img1_byte = (img1 * 255).byte()
            img2_byte = (img2 * 255).byte()

            img1_pre, img2_pre = self.transforms(img1_byte, img2_byte)

            # 3. Run the model and get the flow
            flow = self.model(img1_pre, img2_pre)[-1]
            
            # 4. Return the flow to the original device (so it matches output_prev later)
            return flow.to(original_device)    
            

class DeflickerLoss(nn.Module):
    """
    Combined loss: Temporal Consistency + Reconstruction
    """
    def __init__(self, lambda_rec=1.0):
        super(DeflickerLoss, self).__init__()
        self.lambda_rec = lambda_rec
        self.l1_loss = nn.L1Loss()

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
        # Hopefully to eliminate device shenanigains 
        if occlusion_mask is not None and occlusion_mask.device != output_t.device:
            occlusion_mask = occlusion_mask.to(output_t.device)

        # 1. Reconstruction Loss
        rec_loss = self.l1_loss(output_t, input_t)
        
        # 2. Temporal Loss (Warping Loss)
        # warp_frame will now work because we fixed it to use input.device
        warped_prev = warp_frame(output_prev, flow)
        
        if occlusion_mask is not None:
            temp_diff = torch.abs(output_t - warped_prev) * occlusion_mask
            temp_loss = torch.mean(temp_diff)
        else:
            temp_loss = self.l1_loss(output_t, warped_prev)
            
        total_loss = temp_loss + (self.lambda_rec * rec_loss)
        
        return total_loss, temp_loss, rec_loss


def load_checkpoint(model, optimizer, checkpoint_path):
    """Takes in (model, optimizer, checkpoint_path) and returns start_epochs.
    """
    checkpoint = torch.load(checkpoint_path)
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

    start_epoch = 0

    if settings.USE_DEVICE_IDS:
        device = torch.device(f"cuda:{settings.DEVICE_IDS[0]}")
    
    print("Initializing dataset")
    dataset = VideoDataset(data_dir)
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=settings.NUM_WORKERS)
    else:
        print("error: no samples found. womp womp.")
        exit()

    print("Initializing model")
    model = DeflickerCNN(input_frames=input_frames, num_res_blocks=num_res_blocks, hidden_channels=hidden_channels)
    model.to(device)

    criterion = DeflickerLoss(lambda_rec=LAMBDA).to(settings.VGG_DEVICE_ID if settings.USE_VGG_DEVICE else device)
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
        start_epoch = load_checkpoint(model, optimizer, previous_model_path)

    if settings.USE_DEVICE_IDS:
        print("Wrapping model with nn.DataParallel")
        model = nn.DataParallel(model, device_ids=settings.DEVICE_IDS)

    raft_model = RAFT(settings.VGG_DEVICE_ID if settings.USE_VGG_DEVICE else device)


    print("Generating mask")
    roi_mask = generate_circle_mask(height=dataset.height, width=dataset.width, device=settings.VGG_DEVICE_ID if settings.USE_VGG_DEVICE else device)

    
    for epoch in range(start_epoch, settings.NUM_EPOCHS):
        start_time = time.time()
        print(f"Epoch {epoch+1} / {settings.NUM_EPOCHS}")
        model.train()
        running_loss = 0.0

        if debug:
            print("Debug ending. Exiting . . .")
            exit()

        for batch_idx, (inputs_curr, inputs_prev) in enumerate(dataloader):
            inputs_curr = inputs_curr.to(device)
            inputs_prev = inputs_prev.to(device)

            optimizer.zero_grad()

            input_frame_t = inputs_curr[:, 6:9, :, :]

            input_frame_prev = inputs_curr[:, 3:6, :, :] 
            input_frame_curr = inputs_curr[:, 6:9, :, :]

            flow = raft_model(input_frame_prev, input_frame_curr)

            output_t = model(inputs_curr)
            output_prev = model(inputs_prev)

            total_loss, t_loss, r_loss = criterion(
                output_t=output_t,
                input_t=input_frame_t,
                output_prev=output_prev.detach(),
                flow_prev_to_curr=flow,
                occlusion_mask=roi_mask
            )

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if batch_idx % 20 == 0:
                print(f"    Batch {batch_idx} | Total Loss: {total_loss.item():.4f} | Temp: {t_loss.item():.4f} | Rec: {r_loss.item():.4f} | Time: {time.time() - start_time:.2f}s")

    
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
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
            torch.save(save_dict, settings.MODEL_SAVE_PATH)
            print(f"Model saved to {settings.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main(
        data_dir=os.path.join('media', 'filtered_training_videos'),
        debug=True
    )
