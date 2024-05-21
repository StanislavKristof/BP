import torch.nn as nn
import timm

class FootballRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = timm.create_model("convnext_base", pretrained=True)
        nf = self.feature_extractor.num_features
        
        self.classifier = nn.Sequential(
            # One dimensional representation of an image
            nn.Flatten(), 
            nn.Linear(nf * 4 * 4, 64), # output = batch size
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 14*3), # output = 14 keypoints of [x, y, vis]
        )
        
    def forward(self, x):
        x = self.feature_extractor.forward_features(x)
        y_hat = self.classifier(x)
        y_hat = y_hat.view(-1, 14, 3)
        # Split into keypoint location part and visibility part
        visibility_y_hat = y_hat[:, :, 2:3]
        keypoints_y_hat = y_hat[:, :, :2]
        return visibility_y_hat, keypoints_y_hat

class FootballHeatmapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = timm.create_model("convnext_large", pretrained=True)
        nf = self.feature_extractor.num_features
        
        self.classifier = nn.Sequential(
            # One dimensional representation of an image
            nn.Flatten(),
            nn.Linear(nf * 4 * 4, 32), # output = batch size
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 14*128*128) # 14 keypoints, 128 by 128 heatmap
        )
        
    def forward(self, x):

        x = self.feature_extractor.forward_features(x)

        y_hat = self.classifier(x)

        y_hat = y_hat.view(-1, 14, 128, 128)
        return y_hat