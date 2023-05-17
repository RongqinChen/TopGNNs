from tqdm import tqdm
from datautils.tu.data_module import TU_DataModule


transform = 'to_TPF'
transform = 'to_TPFwithRings'
tu_datamodule = TU_DataModule(
    'MUTAG', 1, 0, 128, 0, transform,
    {'height': 4, 'max_ring_size': 8},
    train_shuffle=False
)

# For testing, train_loader shuffle should be setted as False
train_dataset = tu_datamodule.train_dataset
train_loader = tu_datamodule.train_loader
for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
    tp_list_1 = train_dataset[idx*128: min(len(train_dataset), (idx+1)*128)]
    tp_list_2 = batch.uncollate()
    for tp1, tp2 in zip(tp_list_1, tp_list_2):
        assert str(tp1.tree_edge_image_list) == str(tp2.tree_edge_image_list)

valid_dataset = tu_datamodule.valid_dataset
valid_loader = tu_datamodule.valid_loader
for idx, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
    tp_list_1 = valid_dataset[idx*128: min(len(valid_dataset), (idx+1)*128)]
    tp_list_2 = batch.uncollate()
    for tp1, tp2 in zip(tp_list_1, tp_list_2):
        assert str(tp1.tree_edge_image_list) == str(tp2.tree_edge_image_list)

test_dataset = tu_datamodule.test_dataset
test_loader = tu_datamodule.test_loader
for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
    tp_list_1 = test_dataset[idx*128: min(len(test_dataset), (idx+1)*128)]
    tp_list_2 = batch.uncollate()
    for tp1, tp2 in zip(tp_list_1, tp_list_2):
        assert str(tp1.tree_edge_image_list) == str(tp2.tree_edge_image_list)
