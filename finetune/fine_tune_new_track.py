# import lightning.pytorch as pl
import os
import time
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import BinaryAdapterWrapper
from data_sets.mouse_8_25 import mouse_8_25
from lightning.pytorch.loggers import WandbLogger
from torchsummary import summary
from torchmetrics import AUROC
from torchmetrics.classification import BinaryMatthewsCorrCoef
from lightning.pytorch.callbacks import LearningRateMonitor
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import datetime
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

        # setting up the metric logger
        self.eval_loss = []
        self.eval_accuracy = []

    def forward(self, seq, target):
        return self.model(seq, target=target)

    def training_step(self, batch, batch_idx):
        # NOTE: Don't know why the training step needs to be written in this way
        # print(f"INSIDE training steps and this the shape {batch[0]}")
        seqs, targets = batch
        # seqs = batch[0][0]
        # targets = batch[0][1]
        # print(f"seqs {seqs.shape} targets {targets.shape}")
        loss, logits = self(seqs, target=targets)

        preds = torch.argmax(logits, dim=1)

        correct_count = torch.sum(preds == targets).item()
        accuracy = correct_count / targets.size(0)

        # self.log('train_loss')
        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        # self.log_dict({'train_loss': loss})
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        seqs, target = batch
        loss, logits = self(seqs, target=target)

        #NOTE: comment out might be useful in the end
        # print(f"logits {logits}")
        preds = torch.argmax(logits, dim=1)

        # Calculate accuracy
        correct_count = torch.sum(preds == target).item()
        accuracy = correct_count / target.size(0)

        # Apply softmax along dimension 1 (columns)
        probabilities = torch.softmax(logits, dim=1)

        # Select the probabilities corresponding to class 1
        class_1_probs = probabilities[:, 1]

        # print(f"positive class prediction {class_1_probs}")
        auroc = self.auroc(class_1_probs, target.int())
        
        mcc = self.matthews_corrcoef(class_1_probs, target)
        metrics = {'val_loss': loss, 'val_auroc': auroc, 'val_mcc': mcc, 'val_accuracy': accuracy}
        self.log_dict(metrics)

        self.eval_loss.append(torch.tensor(loss))
        self.eval_accuracy.append(torch.tensor(accuracy))
        return metrics
    
    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Tidying up the environment variables
if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']


# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"
# set the staging dir
os.environ["WANDB_DATA_DIR"] = "/home/114/zl1943/data/z_li_hon/wandb_stage"

os.environ["WANDB_CACHE_DIR"] = "/home/114/zl1943/data/z_li_hon/wandb_stage_cache"


# Assume you have DataLoader setup
batch_size = 8
cell_type = 'mixed_meso'
os.environ["WANDB_RUN_ID"] = f"{cell_type}_fine_tune_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
train_dataloader = DataLoader(mouse_8_25(cell_type=cell_type, data_class='train'), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(mouse_8_25(cell_type=cell_type, data_class='val'), batch_size=batch_size, shuffle=True)
# lr_monitor = LearningRateMonitor(logging_interval='step')

# Initialize model and trainer
# wandb_logger = WandbLogger(project="Enformer-fine-tune-test")
model = EnformerFineTuneModel('EleutherAI/enformer-official-rough')
# for name, para in model.named_parameters():
#     print('{}: {}'.format(name, para.shape))

# summary(model, (1, 196608, 4), (1))
# wandb_logger.watch(model)

# Hyperparameter tuning
scaling_config = ScalingConfig(
    num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
)
storage_path_z = "/g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune/ray_results"

run_config = RunConfig(
    storage_path=storage_path_z,
    local_dir = storage_path_z,
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="ptl/val_accuracy",
        checkpoint_score_order="max",
    ),
    callbacks=[WandbLoggerCallback(project=f"{cell_type}_fine_tune")],
)

search_space = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8]),
}

num_epochs = 8

num_samples = 10


scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

def train_func():
    
    trainer = pl.Trainer(accelerator="auto", devices="auto", 
                         strategy=RayDDPStrategy(find_unused_parameters=True), 
                         callbacks=[RayTrainReportCallback()],
                         plugins=[RayLightningEnvironment()],enable_progress_bar=False,
                         log_every_n_steps=10)  # Adjust settings as required
    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_dataloader, val_dataloader)

ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
)

def tune_func(num_samples=3):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,

        param_space={'train_loop_config': search_space},
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=1,
        ),
    )
    return tuner.fit()

ray.init()
time.sleep(30)
# trainable_with_cpu_gpu = tune.with_resources(trainable, {"cpu": 2, "gpu": 1})
results = tune_func(num_samples=num_samples)
end = results.get_best_result(metric="ptl/val_accuracy", mode="max")
print("BEST END RESULT")
print(end)
# , logger=wandb_logger
# tuner = pl.tuner.tuning.Tuner(trainer)

# lr_finder = tuner.lr_find(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# print(f"lr_finder results  {lr_finder.results}")

# fig = lr_finder.plot(suggest=True)
# fig.savefig("/g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune/lr_finder_plot.png")

# new_lr = lr_finder.suggestion()

# model.hparams.lr = new_lr



# NOTE: 
# /g/data/zk16/zelun/miniconda3/envs/enformer-fine-tune/lib/python3.8/site-packages/torch/nn/modules/conv.py -> this file is mod
# https://colab.research.google.com/github/ray-project/ray/blob/master/doc/source/tune/examples/tune-pytorch-lightning.ipynb
