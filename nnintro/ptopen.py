import torch
import hydra
from omegaconf import DictConfig
import trainer as tr
from pathlib import Path
import numpy as np
import metrics

@hydra.main(version_base=None, config_path="/workspace/nnintro/tools/config", config_name="default")
def my_app(cfg : DictConfig) -> None:
    cfg = {'train': {'batch_size': 64, 'num_workers': 1, 'num_epochs': 1, 'lr': 0.0005}, 
            'exp': {'name': 'newdrconvnext_basetest', 'ver': 0, 'id': '${.name}.${.ver}'}, 
            'dataset': {'final': 
                        {'_target_': 'nnintro.datamodule.FOOTBALL', 'dataset_path': '/mnt/nfs-data/public/xkristof', 'split': None, 'download': False}},
                          'model': {'_target_': 'nnintro.model.FootballRegressor'}
    }
    cfg = DictConfig(cfg)

    trainer = tr.Trainer(cfg)
    model = trainer.model
    file = Path("/mnt/nfs-data/public/xkristof/newdrconvnext_base.pt")
    checkpoint = torch.load(
                file.as_posix(),
                map_location=torch.device("cpu")
    )
    model.load_state_dict(checkpoint["model"])
    trainer.setup()
    trainer.traverse()
    cfg = {'train': {'batch_size': 32, 'num_workers': 1, 'num_epochs': 1, 'lr': 0.00025}, 
            'exp': {'name': 'newhmconvnext_largetest', 'ver': 0, 'id': '${.name}.${.ver}'}, 
            'dataset': {'final': 
                        {'_target_': 'nnintro.datamodule.FOOTBALL_HEATMAP', 'dataset_path': '/mnt/nfs-data/public/xkristof', 'split': None, 'download': False}}, 
                        'model': {'_target_': 'nnintro.model.FootballHeatmapper'}
    }

    cfg = DictConfig(cfg)

    trainer = tr.Trainer(cfg)
    model = trainer.model
    file = Path("/mnt/nfs-data/public/xkristof/newhmconvnext_large.pt")
    checkpoint = torch.load(
                file.as_posix(),
                map_location=torch.device("cpu")
    )
    model.load_state_dict(checkpoint["model"])
    trainer.setup()
    trainer.traverse()
    
    predDR = np.load("/mnt/nfs-data/public/xkristof/newdrconvnext_base0.npy")
    predHM = np.load("/mnt/nfs-data/public/xkristof/newhmconvnext_large0.npy")

    predDRKeypoints, predHMKeypoints, keypoints = metrics.removeTrain(predDR, predHM)
    print("Direct Regression:")
    print("PCK (%):", 100*metrics.PCK(predDRKeypoints, keypoints, 0.15))
    print("PCP (%):", 100*metrics.PCP(predDRKeypoints, keypoints))
    print("PDJ (%):", 100*metrics.PDJ(predDRKeypoints, keypoints, 0.15))
    print("precision:", 100*metrics.get_precision(predDRKeypoints, keypoints, 0.15))
    print("recall:", 100*metrics.get_recall(predDRKeypoints, keypoints, 0.15))

    print(end=2*"\n")
    print("Heatmap Regression:")

    print("PCK (%):", 100*metrics.PCK(predHMKeypoints, keypoints, 0.15))
    print("PCP (%):", 100*metrics.PCP(predHMKeypoints, keypoints))
    print("PDJ (%):", 100*metrics.PDJ(predHMKeypoints, keypoints, 0.15))
    print("precision:", 100*metrics.get_precision(predHMKeypoints, keypoints, 0.15))
    print("recall:", 100*metrics.get_recall(predHMKeypoints, keypoints, 0.15))

my_app()