import os
import hydra

from omegaconf import DictConfig, OmegaConf

from nnintro.trainer import Trainer

from nnintro.loggers import CSVLogger, ModelCheckpointer, WandBLogger


@hydra.main(version_base=None, config_path="config", config_name="default")
def do_main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    trainer = Trainer(cfg)
    # Nejake logovanie alebo take daco ...
    trainer.setup(loggers=[
        CSVLogger(filename=trainer.output_path / "log.csv"),
        ModelCheckpointer(trainer=trainer),
        WandBLogger(
            trainer=trainer,
            project="nnintro",
            entity="stanislavkristof"
        )
    ])

    trainer.train()



if __name__ == "__main__":
    do_main()
