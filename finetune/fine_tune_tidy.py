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
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef, BinaryConfusionMatrix
from lightning.pytorch.callbacks import LearningRateMonitor
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bohb import TuneBOHB
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
        self.matthews_corrcoef = BinaryMatthewsCorrCoef()
        self.f1_score = BinaryF1Score()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.aupr = BinaryAveragePrecision()
        self.cfm = BinaryConfusionMatrix()
        self.auroc = BinaryAUROC()

        # setting up the metric logger
        self.eval_loss = []
        self.eval_accuracy = []
        self.eval_probs = []
        self.eval_target = []
        self.eval_preds = []

    def forward(self, seq, target):
        return self.model(seq, target=target)

    def training_step(self, batch, batch_idx):
        seqs, targets = batch
        loss, logits = self(seqs, target=targets)

        preds = torch.argmax(logits, dim=1)

        correct_count = torch.sum(preds == targets).item()
        accuracy = correct_count / targets.size(0)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        seqs, target = batch
        loss, logits = self(seqs, target=target)

        preds = torch.argmax(logits, dim=1)

        # Calculate accuracy
        correct_count = torch.sum(preds == target).item()
        accuracy = correct_count / target.size(0)

        # Apply softmax along dimension 1 (columns)
        probabilities = torch.softmax(logits, dim=1)

        # Select the probabilities corresponding to class 1
        class_1_probs = probabilities[:, 1]
        

        auroc = self.auroc(class_1_probs, target.int())
        # TODO: add in F1 score, AUPR score precision and recall

        # TODO: need to check if this is correct
        f1_score = self.f1_score(preds, target.int())
        
        precision = self.precision(preds, target.int())

        recall = self.recall(preds, target.int())

        # NOTE: AUPR needs the raw values
        aupr = self.aupr(class_1_probs, target.int())
        
        mcc = self.matthews_corrcoef(class_1_probs, target)
        metrics = {
            'val_loss': loss, 
            'val_auroc': auroc, 
            'val_mcc': mcc, 
            'val_accuracy': accuracy,
            'val_f1': f1_score,
            'val_precision': precision,
            'val_recall': recall,
            'val_aupr': aupr
        }
        self.log_dict(metrics, sync_dist=True)

        self.eval_loss.append(torch.tensor(loss))
        self.eval_accuracy.append(torch.tensor(accuracy))
        self.eval_probs.append(torch.tensor(class_1_probs))
        self.eval_target.append(torch.tensor(target.int()))
        self.eval_preds.append(preds)
        return metrics
    
    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        probs = torch.cat(self.eval_probs)
        targets = torch.cat(self.eval_target)
        preds = torch.cat(self.eval_preds)

        auroc = self.auroc(probs, targets.int())
        f1_score = self.f1_score(preds, targets.int())
        
        precision = self.precision(preds, targets.int())

        recall = self.recall(preds, targets.int())

        aupr = self.aupr(probs, targets.int())
        
        mcc = self.matthews_corrcoef(probs, targets.int())

        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.log("ptl/val_auroc", auroc, sync_dist=True)
        self.log("ptl/val_f1_score", f1_score, sync_dist=True)
        self.log("ptl/val_precision", precision, sync_dist=True)
        self.log("ptl/val_recall", recall, sync_dist=True)
        self.log("ptl/val_aupr", aupr, sync_dist=True)
        self.log("ptl/val_mcc", mcc, sync_dist=True)

        self.eval_loss.clear()
        self.eval_accuracy.clear()
        self.eval_probs.clear()
        self.eval_target.clear()
        self.eval_preds.clear()

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
os.environ["WANDB_MODE"] = "offline"

# set the staging dir
os.environ["WANDB_DATA_DIR"] = "/home/114/zl1943/data/z_li_hon/wandb_stage"

os.environ["WANDB_CACHE_DIR"] = "/home/114/zl1943/data/z_li_hon/wandb_stage_cache"


# Assume you have DataLoader setup
batch_size = 8
cell_type = 'mixed_meso'
os.environ["WANDB_RUN_ID"] = f"{cell_type}_fine_tune_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
train_dataloader = DataLoader(mouse_8_25(cell_type=cell_type, data_class='train'), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(mouse_8_25(cell_type=cell_type, data_class='val'), batch_size=batch_size, shuffle=True)

# Initialize model and trainer
model = EnformerFineTuneModel('EleutherAI/enformer-official-rough')

# Hyperparameter tuning
scaling_config = ScalingConfig(
    num_workers=1, use_gpu=True, resources_per_worker={"CPU": 8, "GPU": 4}
)
storage_path_z = "/g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune/ray_results"

run_config = RunConfig(
    storage_path=storage_path_z,
    local_dir = storage_path_z,
    checkpoint_config=CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="ptl/val_accuracy",
        checkpoint_score_order="max",
    ),
    callbacks=[WandbLoggerCallback(project=f"{cell_type}_fine_tune")],
)

search_space = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([8, 16, 32]),
}


num_epochs = 2

num_samples = 1

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

bohb_search = TuneBOHB(
    metric="ptl/val_loss", mode="min"
)


def tune_func(num_samples=3):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,

        param_space={'train_loop_config': search_space},
        tune_config=tune.TuneConfig(
            metric="ptl/val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=2,
            search_alg=bohb_search,
        ),
    )
    return tuner.fit()

ray.init()
time.sleep(30)
results = tune_func(num_samples=num_samples)
end = results.get_best_result(metric="ptl/val_accuracy", mode="max")
print("BEST END RESULT")
print(end)