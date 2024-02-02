import torch
# from enformer_pytorch import Enformer
# from enformer_pytorch.finetune import HeadAdapterWrapper

# enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough')

# model = HeadAdapterWrapper(
#     enformer = enformer,
#     num_tracks = 128,
#     post_transformer_embed = False   # by default, embeddings are taken from after the final pointwise block w/ conv -> gelu - but if you'd like the embeddings right after the transformer block with a learned layernorm, set this to True
# ).cuda()

# seq = torch.randint(0, 5, (1, 196_608 // 2,)).cuda()
# target = torch.randn(1, 200, 128).cuda()  # 128 tracks

# print(f"seq structure {seq.shape}")

# loss = model(seq, target = target)
# loss.backward()