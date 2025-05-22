import os.path as osp 
path = 'legged_gym/resources/motions/h1/stable_punch.pkl' 
if osp.isfile(path): 
    print(f"{path} 是一个文件") 
else: 
    print(f"{path} 不是一个文件")