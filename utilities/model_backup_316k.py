"""
Neural Network Architecture for Double DQN (DDQN) - VERSION 2
Improved architecture with Global Average Pooling for 100% image coverage

Key Improvements from V1:
1. 5 convolutional layers (was 3) for better feature extraction
2. Global Average Pooling for guaranteed 100% image coverage
3. Much smaller FC layers (16K params vs 6.4M) - prevents overfitting
4. Total params: ~316K (was 6.5M) - 20× reduction!
5. Batch normalization for training stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network with Global Average Pooling
    
    Input: (batch, 4, 84, 252) - 4 stacked grayscale frames
    Output: (batch, 3) - Q-values for [nothing, jump, duck]
    
    Architecture Philosophy:
    - Deeper CNN (5 layers) extracts hierarchical spatial features
    - Global Average Pooling ensures every feature sees 100% of image
    - Small FC layers (128→128→3) prevent overfitting
    - Batch norm stabilizes training
    
    Total Parameters: ~316,323 (vs 6.5M in old version!)
    """
    
    def __init__(self, n_frames=4, n_actions=3):
        super(DQN, self).__init__()
        
        # ========================
        # CONVOLUTIONAL LAYERS
        # ========================
        # Progressive feature extraction from pixels to high-level concepts
        
        # Conv1: Learn basic edges and textures
        # Input: (4, 84, 252) → Output: (32, 20, 62)
        self.conv1 = nn.Conv2d(n_frames, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv2: Learn mid-level patterns (shapes, corners)
        # Input: (32, 20, 62) → Output: (64, 9, 30)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv3: Learn higher-level features (object parts)
        # Input: (64, 9, 30) → Output: (64, 7, 28)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Conv4: Learn complex patterns (obstacle types)
        # Input: (64, 7, 28) → Output: (128, 5, 26)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Conv5: Learn game-specific concepts (danger zones, safe paths)
        # Input: (128, 5, 26) → Output: (128, 3, 24)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        # ========================
        # GLOBAL AVERAGE POOLING
        # ========================
        # CRITICAL: This is what gives us 100% image coverage!
        # Each of the 128 channels becomes a single number (average over all spatial positions)
        # Result: (128, 3, 24) → (128, 1, 1) → flattens to 128 features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ========================
        # FULLY CONNECTED LAYERS
        # ========================
        # Small FC layers - 128 features is plenty for jump/duck/nothing decision
        
        # FC1: Process the 128 global features
        self.fc1 = nn.Linear(128, 128)
        
        # FC2: Output Q-values for 3 actions
        self.fc2 = nn.Linear(128, n_actions)
        
        # Dropout for regularization (optional, can be removed if overfitting not an issue)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: (batch, 4, 84, 252) - 4 stacked frames
        
        Returns:
            Q-values: (batch, 3) - Q(s, nothing), Q(s, jump), Q(s, duck)
        """
        # Convolutional feature extraction with batch norm
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 32, 20, 62)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 64, 9, 30)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 64, 7, 28)
        x = F.relu(self.bn4(self.conv4(x)))  # (batch, 128, 5, 26)
        x = F.relu(self.bn5(self.conv5(x)))  # (batch, 128, 3, 24)
        
        # Global Average Pooling - 100% image coverage!
        x = self.global_pool(x)              # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)            # (batch, 128) - flatten
        
        # Fully connected layers
        x = F.relu(self.fc1(x))              # (batch, 128)
        x = self.dropout(x)                  # Regularization
        x = self.fc2(x)                      # (batch, 3) - Q-values
        
        return x
    
    def get_feature_vector(self, x):
        """
        Extract the 128-dimensional feature vector (useful for visualization/analysis)
        
        Args:
            x: (batch, 4, 84, 252) - 4 stacked frames
        
        Returns:
            features: (batch, 128) - global feature vector
        """
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        return x


if __name__ == "__main__":
    # Test the network
    print("="*70)
    print("TESTING NEW DQN ARCHITECTURE WITH GLOBAL AVERAGE POOLING")
    print("="*70)
    
    model = DQN(n_frames=4, n_actions=3)
    print(f"\nModel architecture:\n{model}\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*70)
    print("PARAMETER BREAKDOWN:")
    print("="*70)
    
    conv_params = 0
    bn_params = 0
    fc_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        if 'conv' in name:
            conv_params += num_params
        elif 'bn' in name:
            bn_params += num_params
        elif 'fc' in name:
            fc_params += num_params
        print(f"{name:30s}: {num_params:>10,}")
    
    print("="*70)
    print(f"Conv layers:        {conv_params:>10,} ({conv_params/total_params*100:.1f}%)")
    print(f"Batch norm:         {bn_params:>10,} ({bn_params/total_params*100:.1f}%)")
    print(f"FC layers:          {fc_params:>10,} ({fc_params/total_params*100:.1f}%)")
    print(f"Total parameters:   {total_params:>10,}")
    print(f"Trainable params:   {trainable_params:>10,}")
    print("="*70)
    
    # Test forward pass
    print("\nTESTING FORWARD PASS:")
    print("="*70)
    
    test_input = torch.randn(2, 4, 84, 252)  # Batch of 2
    print(f"Input shape:        {tuple(test_input.shape)}")
    
    output = model(test_input)
    print(f"Output shape:       {tuple(output.shape)}")
    print(f"\nSample Q-values for first batch:")
    print(f"  Q(nothing) = {output[0, 0].item():.4f}")
    print(f"  Q(jump)    = {output[0, 1].item():.4f}")
    print(f"  Q(duck)    = {output[0, 2].item():.4f}")
    
    # Test feature extraction
    features = model.get_feature_vector(test_input)
    print(f"\nFeature vector shape: {tuple(features.shape)}")
    print(f"Each state represented by {features.shape[1]} global features")
    
    print("\n" + "="*70)
    print("COMPARISON WITH OLD ARCHITECTURE:")
    print("="*70)
    print(f"Old model:  6,502,563 parameters")
    print(f"New model:    {total_params:,} parameters")
    print(f"Reduction:  {6502563/total_params:.1f}× smaller!")
    print(f"\nOld coverage: 14% of image width")
    print(f"New coverage: 100% guaranteed (global pooling)")
    print("="*70)
