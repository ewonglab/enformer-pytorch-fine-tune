import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import BinaryAdapterWrapper
from data_sets.mouse_8_25 import mouse_8_25
from lightning.pytorch.loggers import WandbLogger
from torchsummary import summary
from torchmetrics import AUROC
from torchmetrics.classification import BinaryMatthewsCorrCoef

# 1. Poisson negative log-likelihood
# 2. 

# from pytorch_lightning.callbacks import Callback

# class MetricLogger(pl.Callback):

#     def on_epoch_end(self, trainer, pl_module):
#         metrics = {
#             'epoch_loss': pl_module.epoch_loss.item(),
#             'epoch_accuracy': pl_module.epoch_accuracy.item(),
#             'epoch_auroc': pl_module.epoch_auroc.item(),
#             'epoch_mcc': pl_module.epoch_mcc.item()
#         }
#         trainer.logger.experiment.log(metrics)
    
#     def on_train_end(self, trainer, pl_module):
#         # If you calculate final metrics for the entire training phase, you can log them here
#         # Example:
#         metrics = {
#             'final_loss': pl_module.final_loss.item(),
#             'final_accuracy': pl_module.final_accuracy.item(),
#             'final_auroc': pl_module.final_auroc.item(),
#             'final_mcc': pl_module.final_mcc.item()
#         }
#         trainer.logger.experiment.log(metrics)

#     def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
#         return super().on_validation_end(trainer, pl_module)
    

class EnformerFineTuneModel(pl.LightningModule):
    def __init__(self, pretrained_model_name):
        super(EnformerFineTuneModel, self).__init__()
        
        # NOTE: use_checkpointing is to reduce memory usage
        self.enformer = Enformer.from_pretrained(pretrained_model_name, use_checkpointing = True)
        self.model = BinaryAdapterWrapper(
            enformer = self.enformer,
            num_tracks = 2,
            post_transformer_embed = False,
            auto_set_target_length = False
        )
        # Define your loss function here

        # lr set (default value)
        self.lr = 1e-3

        # logging the metric used
        self.auroc = AUROC(task="binary") 
        self.matthews_corrcoef = BinaryMatthewsCorrCoef()

    def forward(self, seq, target):
        return self.model(seq, target=target)

    def training_step(self, batch, batch_idx):
        # NOTE: Don't know why the training step needs to be written in this way
        # print(f"INSIDE training steps and this the shape {batch[0]}")
        seqs, targets = batch
        # seqs = batch[0][0]
        # targets = batch[0][1]
        # print(f"seqs {seqs.shape} targets {targets.shape}")
        loss, _ = self(seqs, target=targets)
        # self.log('train_loss')
        self.log_dict({'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        seqs, target = batch
        loss, logits = self(seqs, target=target)

        #NOTE: comment out might be useful in the end
        print(f"logits {logits}")
        preds = torch.argmax(logits, dim=1)
        
        # Apply softmax along dimension 1 (columns)
        probabilities = torch.softmax(logits, dim=1)

        # Select the probabilities corresponding to class 1
        class_1_probs = probabilities[:, 1]

        print(f"positive class prediction {class_1_probs}")
        auroc = self.auroc(class_1_probs, target.int())
        
        mcc = self.matthews_corrcoef(class_1_probs, target)
        metrics = {'val_loss': loss, 'val_auroc': auroc, 'val_mcc': mcc}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer





# Assume you have DataLoader setup
batch_size = 8
train_dataloader = DataLoader(mouse_8_25(data_class='train'), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(mouse_8_25(data_class='val'), batch_size=batch_size, shuffle=True)

# Initialize model and trainer
# wandb_logger = WandbLogger(project="Enformer-fine-tune-test")
model = EnformerFineTuneModel('EleutherAI/enformer-official-rough')
# for name, para in model.named_parameters():
#     print('{}: {}'.format(name, para.shape))

# summary(model, (1, 196608, 4), (1))
# wandb_logger.watch(model)
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=2, log_every_n_steps=16)  # Adjust settings as required
# , logger=wandb_logger
# tuner = pl.tuner.tuning.Tuner(trainer)

# lr_finder = tuner.lr_find(model, train_dataloaders=[train_dataloader], val_dataloaders=[val_dataloader])

# print(f"lr_finder results  {lr_finder.results}")

# fig = lr_finder.plot(suggest=True)
# fig.savefig("/g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune/lr_finder_plot.png")

# new_lr = lr_finder.suggestion()

# model.hparams.lr = new_lr


trainer.fit(model, train_dataloader, val_dataloader)


# NOTE: 
# /g/data/zk16/zelun/miniconda3/envs/enformer-fine-tune/lib/python3.8/site-packages/torch/nn/modules/conv.py -> this file is mod
