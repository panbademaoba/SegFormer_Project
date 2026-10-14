from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class OpenEarthMapDataset(BaseSegDataset):

    METAINFO = dict(
        classes=(
            "background",
            "bareland",
            "rangeland",
            "developed_space",
            "road",
            "tree",
            "water",
            "agriculture",
            "building",
        )
    )


    def load_data_list(self):

        data_list = []


        with open(
            self.ann_file,
            "r",
            encoding="utf-8"
        ) as f:

            for idx, line in enumerate(f):

                img_path = line.strip()

                if not img_path:
                    continue


                img_path = img_path.replace(
                    "\\",
                    "/"
                )


                seg_path = img_path.replace(
                    "/images/",
                    "/labels/"
                )


                data_list.append(
                    dict(
                        img_path=img_path,
                        seg_map_path=seg_path,
                        reduce_zero_label=False,

                        seg_fields=[]
                    )
                )


        return data_list