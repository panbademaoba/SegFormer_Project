from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


register_all_modules()


cfg = Config.fromfile(
    r"D:\SegFormer_Project\configs\openearthmap\segformer_b0_openearthmap.py"
)


model = MODELS.build(cfg.model)


print(type(model))