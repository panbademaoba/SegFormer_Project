from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules


register_all_modules()



cfg = Config.fromfile(

    r"D:\SegFormer_Project\configs\openearthmap\segformer_b0_openearthmap.py"

)



cfg.train_cfg.max_iters = 10



cfg.val_dataloader = None
cfg.val_cfg = None
cfg.val_evaluator = None



cfg.test_dataloader = None
cfg.test_cfg = None
cfg.test_evaluator = None



cfg.work_dir = (

    r"D:\SegFormer_Project\work_dirs\R3_test_train"

)



runner = Runner.from_cfg(cfg)



runner.train()