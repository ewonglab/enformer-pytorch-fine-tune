import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import BinaryAdapterWrapper
from data_sets.mouse_8_25 import mouse_8_25
from lightning.pytorch.loggers import WandbLogger
from torchsummary import summary


# 1. Poisson negative log-likelihood
# 2. 

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
        # TODO: Change the loss function here

    def forward(self, seq, target):
        return self.model(seq, target=target)

    def training_step(self, batch, batch_idx):
        seqs, targets = batch
        print(f"seqs {seqs.shape} targets {targets.shape}")
        outputs = self(seqs, target=targets)
        self.log('train_loss', outputs)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Assume you have DataLoader setup
batch_size = 1
train_dataloader = DataLoader(mouse_8_25(data_class='train'), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(mouse_8_25(data_class='val'), batch_size=batch_size, shuffle=True)

# Initialize model and trainer
# wandb_logger = WandbLogger(project="Enformer-fine-tune-tesst")
model = EnformerFineTuneModel('EleutherAI/enformer-official-rough')
# for name, para in model.named_parameters():
#     print('{}: {}'.format(name, para.shape))

# summary(model, (1, 196608, 4), (1))
# wandb_logger.watch(model)
torch.cuda.empty_cache()
print(f"Memory Allocated {torch.cuda.memory_allocated()}")
print(f"Memory Reserved {torch.cuda.memory_reserved()}")
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=2)  # Adjust settings as required
trainer.fit(model, train_dataloader, val_dataloader)
