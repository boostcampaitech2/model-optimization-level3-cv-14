import yaml
import os
from shutil import copyfile

with open('./configs/data/taco.yaml') as f:
        data_config = yaml.load(f,yaml.SafeLoader)

data_path = data_config["DATA_PATH"]

train_path = os.path.join(data_path, "train")
save_path = os.path.join(data_path,"train_split")

category_list = ["Metal","Paper","Paperpack","Plastic","Plasticbag","Styrofoam"]

for cat in category_list:
    cat_src = os.path.join(train_path,cat)
    cat_dst = os.path.join(save_path,cat)
    img_list = os.listdir(cat_src)
    print(cat,len(os.listdir(cat_src)))
    for i in range(len(img_list)):
        if i%10>0: continue
        img = img_list[i]
        img_src = os.path.join(cat_src,img)
        img_dst = os.path.join(cat_dst,img)
        copyfile(img_src,img_dst)
    print(cat,len(os.listdir(cat_dst)))
