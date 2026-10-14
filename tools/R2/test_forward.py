import torch

from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


register_all_modules()


cfg = Config.fromfile(
    r"D:\SegFormer_Project\configs\openearthmap\segformer_b0_openearthmap.py"
)


model = MODELS.build(cfg.model)

model.eval()


x = torch.randn(
    1,
    3,
    512,
    512
)


with torch.no_grad():

    result = model(
        x,
        mode='tensor'
    )


print(result.shape)