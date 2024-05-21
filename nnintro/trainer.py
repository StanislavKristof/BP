
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from nnintro.datamodule import DataModule
import nnintro.datamodule
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from nnintro.utils import get_progress
from torchmetrics.classification import BinaryAccuracy
import numpy as np
from torchsummary import summary
from math import ceil

from nnintro.loggers import Loggers
from nnintro.utils import Statistics

class Trainer:
    def __init__(self, cfg):
        self.hparams = cfg
        self.datamodule = DataModule(self.hparams)
        self.output_path = Path(HydraConfig.get().run.dir)

        # CUDA / CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create our model
        self.model = instantiate(cfg.model)

        # Create optimizer
        self.opt = optim.Adam(
            params=self.model.classifier.parameters(),
            lr=self.hparams.train.lr
        )

        # Create loss function
        self.keypointLoss = nn.MSELoss() 
        self.visibilityLoss = nn.BCEWithLogitsLoss()
        self.visibilityAccuracy = BinaryAccuracy().to(self.device)


        # Compute using GPU
        self.model = self.model.to(self.device)
        # Display model info
        summary(self.model, input_size=(3,128,128)) # channel number, image dimension


    def setup(self, loggers=[]):
        # Setup datasets and loaders
        self.datamodule.setup()
        try: # if TRAIN or VAL
            if issubclass(type(self.datamodule.ds_train), nnintro.datamodule.FOOTBALL_HEATMAP):
                self.method = "heatmap"
            if issubclass(type(self.datamodule.ds_train), nnintro.datamodule.FOOTBALL):
                self.method = "regression"
        except AttributeError: # if TRAVERSE
            if issubclass(type(self.datamodule.ds_final), nnintro.datamodule.FOOTBALL_HEATMAP):
                self.method = "heatmap"
            if issubclass(type(self.datamodule.ds_final), nnintro.datamodule.FOOTBALL):
                self.method = "regression"
        self.log = Loggers(loggers=loggers)

    def traverse(self):
        stats = Statistics()
        self.traverse_epoch(stats)

    # Convert visibility logits to either 1 (visible) or 0 (non-visible)
    def rounded_sigmoid(self, x):
        return np.round(1 / (1 + np.exp(-x)))
    
    def convertHeatmapsToKeypoint(self, y_hat):
        collect = list()
        for pose in range(y_hat.shape[0]):
            keypoints = []
            for i in range(y_hat.shape[1]): # 14
                kernel = y_hat[pose][i]
                visibility = 1
                if np.all(kernel == 0):
                    visibility = 0

                y_max = ceil(kernel.argmax() / y_hat.shape[2]) + 1
                x_max = ceil(kernel.argmax() % y_hat.shape[3]) + 1

                x = x_max / y_hat.shape[3]
                y = y_max / y_hat.shape[2]

                keypoint = [x, y, visibility]
                keypoints.append(keypoint)
            keypoints = np.array(keypoints)
            collect.append(keypoints)
        return collect

    def traverse_epoch(self, stats):
        # List of predictions, collects all predictions, to create an array
        y_hat_list = list()
        self.model.train()
        progress = get_progress(self.datamodule.loader_final, name=f"TRAVERSE")

        if self.method == "regression":
            for batch in progress:
                
                # Split into images and classes
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                # Split y into visibility part and keypoints location part
                visibility_y = y[:, :, 2:3] 
                keypoints_y = y[:, :, :2]

                # Compute prediction
                visibility_y_hat, keypoints_y_hat = self.model(x)
                combination_y_hat = torch.cat((keypoints_y_hat, visibility_y_hat), dim=2)
                
                # Add prediction to list of predictions
                y_hat_list.append(combination_y_hat)

                # Compute loss
                # keypoint is not visible, the model will not be punished
                keyloss = self.keypointLoss(keypoints_y_hat * visibility_y, keypoints_y * visibility_y)
                visloss = self.visibilityLoss(visibility_y_hat, visibility_y)

                # Compute accuracy
                visacc = self.visibilityAccuracy(
                    torch.sigmoid(visibility_y_hat), 
                    visibility_y                             
                    )

                # Backprop and optimize
                self.opt.zero_grad()
                loss = visloss + keyloss
                loss.backward()
                self.opt.step()

                # Statistics
                stats.step("keyloss", keyloss.item())
                stats.step("visloss", visloss.item())
                stats.step("mse", keyloss.item())
                stats.step("bin", visacc.item())
                progress.set_postfix(stats.get_stats())
            
            # Convert list to tensor
            y_hat_array = torch.cat(y_hat_list)
            y_hat_array = torch.Tensor.cpu(y_hat_array)
            y_hat_array = y_hat_array.detach().numpy()

            # Convert visibility logits to 1 or 0
            y_hat_array[:, :, 2] = self.rounded_sigmoid(y_hat_array[:,  :, 2])

            # Save the tensor, so that it can be measured and visualised
            np.save("/mnt/nfs-data/public/xkristof/newdrconvnext_base0", y_hat_array)
        
        elif self.method == "heatmap":
            for batch in progress:
                # Split into images and classes
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                y = y.double()

                # Compute prediction and loss
                y_hat = self.model(x)
                y_hat = y_hat.double()
                keyloss = self.keypointLoss(y_hat, y)

                # Backprop and optimize
                self.opt.zero_grad()
                loss = keyloss
                loss.backward()
                self.opt.step()

                # Statistics
                stats.step("mse", keyloss.item())
                
                progress.set_postfix(stats.get_stats())
                y_hat = torch.Tensor.cpu(y_hat)
                y_hat = y_hat.detach().numpy()

                # Convert heatmap into a [x, y, vis] tensor
                collection = self.convertHeatmapsToKeypoint(y_hat)
                y_hat_list = y_hat_list + collection
                """
                print("shape", y_hat.shape)
                for pose in range(y_hat.shape[0]):
                    keypoints = []
                    for i in range(y_hat.shape[1]): # 14
                        kernel = y_hat[pose][i]
                        visibility = 1
                        if np.all(kernel == 0):
                            visibility = 0

                        y_max = ceil(kernel.argmax() / y_hat.shape[2]) + 1
                        x_max = ceil(kernel.argmax() % y_hat.shape[3]) + 1

                        x = x_max / y_hat.shape[3]
                        y = y_max / y_hat.shape[2]

                        keypoint = [x, y, visibility]
                        keypoints.append(keypoint)
                    keypoints = np.array(keypoints)
                    y_hat_list.append(keypoints)
                """
            y_hat_array = np.array(y_hat_list)
            np.save("/mnt/nfs-data/public/xkristof/newhmconvnext_large0.npy", y_hat_array)


    def train(self):
        # Main training loop
        self.log.on_training_start()
        
        for epoch in range(self.hparams.train.num_epochs):
            stats = Statistics()

            self.train_epoch(epoch, stats)
            self.validate_epoch(epoch, stats)

            # Log statistics
            self.log.on_epoch_end(epoch, stats.get_stats())

        # Save the model
        self.log.on_training_end()

    def train_epoch(self, epoch, stats):
        print(f"Training epoch: {epoch}")

        # Training phase
        self.model.train()
        progress = get_progress(self.datamodule.loader_train, name=f"{epoch}")

        for batch in progress:
            
            # Split into images and classes
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            if self.method == "regression":
                # Split y into visibility part and keypoints location part
                visibility_y = y[:, :, 2:3]
                keypoints_y = y[:, :, :2]

                # Compute prediction
                visibility_y_hat, keypoints_y_hat = self.model(x)

                # Compute loss
                # keypoint is not visible, the model will not be punished
                keyloss = self.keypointLoss(keypoints_y_hat * visibility_y, keypoints_y * visibility_y)
                visloss = self.visibilityLoss(visibility_y_hat, visibility_y)
                
                # Compute accuracy
                visacc = self.visibilityAccuracy(
                    torch.sigmoid(visibility_y_hat), 
                    visibility_y                             
                )

                # Backprop and optimize
                self.opt.zero_grad()
                loss = visloss + keyloss
                loss.backward()
                self.opt.step()

                # Statistics
                stats.step("visloss", visloss.item())
                stats.step("mse", keyloss.item())
                stats.step("bin", visacc.item())
                progress.set_postfix(stats.get_stats())

            if self.method == "heatmap":
                y = y.double()

                # Compute prediction and loss
                y_hat = self.model(x)
                y_hat = y_hat.double()
                keyloss = self.keypointLoss(y_hat, y)

                # Backprop and optimize
                self.opt.zero_grad()
                loss = keyloss
                loss.backward()
                self.opt.step()

                # Statistics
                stats.step("mse", keyloss.item())
                progress.set_postfix(stats.get_stats())



    def validate_epoch(self, epoch, stats):
        print(f"Validating epoch: {epoch}")

        # Validation phase
        self.model.eval()
        progress = get_progress(self.datamodule.loader_val, name=f"{epoch}")

        # Save some computation
        with torch.no_grad():

            for batch in progress:
                
                # Split into images and classes
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
    
                if self.method == "regression":

                    # Split y into visibility part and keypoints location part
                    visibility_y = y[:, :, 2:3]
                    keypoints_y = y[:, :, :2]

                    # Compute prediction
                    visibility_y_hat, keypoints_y_hat = self.model(x)
                    # Compute loss
                    # keypoint is not visible, the model will not be punished
                    keyloss = self.keypointLoss(keypoints_y_hat * visibility_y, keypoints_y * visibility_y)
                    visloss = self.visibilityLoss(visibility_y_hat, visibility_y)
                    
                    # Compute accuracy
                    visacc = self.visibilityAccuracy(
                        torch.sigmoid(visibility_y_hat), 
                        visibility_y                             
                        )

                    # Statistics
                    stats.step("val_visloss", visloss.item())
                    stats.step("val_mse", keyloss.item())
                    stats.step("val_bin", visacc.item())

                    progress.set_postfix(stats.get_stats())

                if self.method == "heatmap":
                    y = y.double()

                    # Compute prediction and loss
                    y_hat = self.model(x)
                    y_hat = y_hat.double()
                    keyloss = self.keypointLoss(y_hat, y)

                    # Statistics
                    stats.step("val_mse", keyloss.item())
                    progress.set_postfix(stats.get_stats())


    def save_checkpoint(self, file_name):
        checkpoint = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict()
        }
        torch.save(checkpoint, file_name.as_posix())
