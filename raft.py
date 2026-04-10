#!/usr/bin/env python3


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
            img1_byte = (torch.clamp(img1, 0, 1) * 255).byte()
            img2_byte = (torch.clamp(img2, 0, 1) * 255).byte()

            img1_pre, img2_pre = self.transforms(img1_byte, img2_byte)

            flow = self.model(img1_pre, img2_pre)[-1]
            
            return flow
