import os

path = r"D:\OpenEarthMap\OpenEarthMap\OpenEarthMap_wo_xBD\tokyo\labels"

print("labels路径:")
print(path)

print("\n里面文件数量:")
files = os.listdir(path)
print(len(files))

print("\n前10个文件:")
print(files[:10])