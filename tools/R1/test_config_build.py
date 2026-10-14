from mmengine.config import Config

import mmseg_custom.openearthmap

from mmseg_custom.openearthmap import OpenEarthMapDataset


cfg = Config.fromfile(
    r"D:\SegFormer_Project\configs\openearthmap\segformer_b0_openearthmap.py"
)


dataset_cfg = cfg.train_dataloader.dataset


dataset = OpenEarthMapDataset(
    ann_file=dataset_cfg.ann_file,
    pipeline=[]
)


print(len(dataset))

print(dataset.get_data_info(0))