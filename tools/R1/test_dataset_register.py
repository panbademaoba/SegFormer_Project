from mmseg.registry import DATASETS

import mmseg_custom.openearthmap


dataset = DATASETS.get("OpenEarthMapDataset")

print(dataset)