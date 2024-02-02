from ray import tune


# Basic Config
batch_size = 8
cell_type = 'mixed_meso'

    
    

# ray related config

num_epochs = 8
num_samples = 10

search_space = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([2, 4, 8]),
}

scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)
