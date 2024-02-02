from data_sets import mouse_8_25
from torch.utils.data import DataLoader

cell_type = 'mixed_meso'
train_dataloader = DataLoader(mouse_8_25(cell_type=cell_type, data_class='train'), shuffle=True, batch_size=config['batch_size'], num_workers=4)



