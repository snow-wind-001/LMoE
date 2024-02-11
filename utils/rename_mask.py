import os

def rename_mask(root_path):
        for root,_,files in os.walk(root_path):
            for filename in files:
                if filename.endswith('.jpg'):
                    newname = filename.split('.jpg')[0]
                    newname = newname.split('mask')[0]
                    new_filename = newname + '_mask'+'.jpg'
                    full_path = os.path.join(root,filename)
                    new_full_path = os.path.join(root,new_filename)
                    os.rename(full_path,new_full_path)
                    print(f"{full_path}重命名为{new_full_path}")
file___path ='/home/user/git_code/cometition/AnomalyGPT/data/mvtec/Working_4/ground_truth'
rename_mask(file___path)



