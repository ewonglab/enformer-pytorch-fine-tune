import lighting.pytroch as pl
import torch
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import HeadAdapterWrapper
from data_sets.mouse_8_25 import mouse_8_25


# 1. Poisson negative log-likelihood
# 2. 
        
class EnformerFineTuneModel(pl.LightningModule):
    def __init__(self, pretrained_model_name):
        super(EnformerFineTuneModel, self).__init__()
        
        self.enformer = Enformer.from_pretrained(pretrained_model_name)
        self.model = HeadAdapterWrapper(
            enformer = self.enformer,
            num_tracks = 2,
            post_transformer_embed = False
        )
        
        # Define your loss function here
        # TODO: Change the loss function here
        self.criterion = torch.nn.MSELoss()  # Example: Mean Squared Error

    def forward(self, seq, target):
        return self.model(seq, target=target)

    def training_step(self, batch, batch_idx):
        seqs, targets = batch
        outputs = self(seqs, target=targets)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Assume you have DataLoader setup
# train_dataloader = DataLoader(YourDataset(), batch_size=your_batch_size, shuffle=True)

# Initialize model and trainer
model = EnformerFineTuneModel('EleutherAI/enformer-official-rough')
trainer = pl.Trainer(max_epochs=10, gpus=1)  # Adjust settings as required
trainer.fit(model, train_dataloader)
