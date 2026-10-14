from mmseg_custom.openearthmap import OpenEarthMapDataset


dataset = OpenEarthMapDataset(
    ann_file=r"D:\SegFormer_Project\work_dirs\R0_audit\train_pool.txt"
)


print(len(dataset))

print(dataset.get_data_info(0))