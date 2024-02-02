import torch
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import HeadAdapterWrapper

enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough')

model = HeadAdapterWrapper(
    enformer = enformer,
    num_tracks = 128,
    post_transformer_embed = False   # by default, embeddings are taken from after the final pointwise block w/ conv -> gelu - but if you'd like the embeddings right after the transformer block with a learned layernorm, set this to True
).cuda()

# TODO: check whether I need to put seq and target onto .cuda()
seq = torch.randint(0, 5, (1, 196_608 // 2,)).cuda()
target = torch.randn(1, 200, 128).cuda()  # 128 tracks

loss = model(seq, target = target)
loss.backward()



# model = MyModelClass(hparams)
# trainer = Trainer()
# tuner = Tuner(trainer)

# # Run learning rate finder
# lr_finder = tuner.lr_find(model)

# # Results can be found in
# print(lr_finder.results)

# # Plot with
# fig = lr_finder.plot(suggest=True)
# fig.show()

# # Pick point based on plot, or get suggestion
# new_lr = lr_finder.suggestion()

# # update hparams of the model
# model.hparams.lr = new_lr

# # Fit model
# trainer.fit(model)