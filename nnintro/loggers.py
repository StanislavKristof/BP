import wandb
import os

class Logger:
    def __init__(self):
        pass

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    def on_epoch_end(self, epoch, stats):
        pass

class Loggers:
    def __init__(self, loggers):
        self.loggers = loggers

    def on_training_start(self):
        for logger in self.loggers:
            logger.on_training_start()

    def on_training_end(self):
        for logger in self.loggers:
            logger.on_training_end()

    def on_epoch_end(self, epoch, stats):
        for logger in self.loggers:
            logger.on_epoch_end(epoch, stats)

class CSVLogger(Logger):
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.lines = 0
        self.separator = ","

    def on_training_start(self):
        self.lines = 0
        #os.chdir("/workspace/nnintro/.scratch/")
        #print(self.filename)
        self.file = open(self.filename, "w")

    def on_training_end(self):
        # is not None
        if self.file != None:
            self.file.close()
            self.file = None

    def on_epoch_end(self, epoch, stats):
        stats["epoch"] = epoch
        # is not None
        if self.file != None:
            if self.lines == 0:
                # insert CSV header
                line = self.separator.join(list(stats.keys()))
                self.file.write(line + "\n")
            
            # Combine all the values
            values = [ "{:.10f}".format(stats[kluc]) for kluc in stats.keys() ]
            line = self.separator.join(values)
            self.file.write(line + "\n")
            self.file.flush() # write to the disk
            self.lines += 1

class ModelCheckpointer(Logger):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_training_end(self):
        self.trainer.save_checkpoint(self.trainer.output_path / "checkpoint.pt")

class WandBLogger(Logger):
    def __init__(self, trainer, project, entity):
        self.trainer = trainer
        self.project = project
        self.entity = entity
        self.run = None

    def on_training_start(self):
        # Initialize WandB
        wandb.config = self.trainer.hparams
        self.run = wandb.init(
            config = self.trainer.hparams,
            project = self.project,
            entity = self.entity,
            name = self.trainer.hparams.exp.id,
            reinit = True
        )


    def on_training_end(self):
        # is not None
        if self.run != None:
            wandb.finish()
            self.run = None

    def on_epoch_end(self, epoch, stats):
        self.run.log(stats)