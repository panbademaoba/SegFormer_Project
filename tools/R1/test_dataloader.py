from mmengine.config import Config
from mmseg.registry import DATASETS

from mmseg.utils import register_all_modules

import mmseg_custom.openearthmap


register_all_modules()


cfg = Config.fromfile(
    r"D:\SegFormer_Project\configs\openearthmap\segformer_b0_openearthmap.py"
)


dataset_cfg = cfg.train_dataloader.dataset


dataset = DATASETS.build(dataset_cfg)


print("dataset size:", len(dataset))


data = dataset[0]


print(type(data))

print(data.keys())