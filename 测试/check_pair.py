import os


ROOT = r"D:\OpenEarthMap\OpenEarthMap\OpenEarthMap_wo_xBD\tokyo"


image_dir = os.path.join(ROOT, "images")
label_dir = os.path.join(ROOT, "labels")


images = set(
    os.listdir(image_dir)
)

labels = set(
    os.listdir(label_dir)
)


print("images数量:", len(images))
print("labels数量:", len(labels))


print("\nimages前10:")
print(list(images)[:10])


print("\nlabels前10:")
print(list(labels)[:10])


print("\n图片独有:")
print(images - labels)


print("\n标签独有:")
print(labels - images)