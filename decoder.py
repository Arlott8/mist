import torch
import torch.nn as nn

class WatermarkDecoder(nn.Module):
    """
    Standard HiDDeN Decoder Architecture.
    Extracts a binary message from an RGB image.
    """
    def __init__(self, message_length=30):
        super(WatermarkDecoder, self).__init__()
        
        self.layers = nn.Sequential(
            # Input: (Batch, 3, H, W) -> Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Conv Block 2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Conv Block 3
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # Downsample
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Conv Block 4
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv Block 5
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Global Average Pooling to get a fixed size feature vector
            # This allows it to work on any image size (256x256, 512x512, etc.)
            nn.AdaptiveAvgPool2d((1, 1)) 
        )

        # Final Linear Layer to predict the bits
        self.linear = nn.Linear(64, message_length)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1) # Flatten: (Batch, 64)
        x = self.linear(x)        # Linear: (Batch, Message_Length)
        return x