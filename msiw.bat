python tools/train_UDA.py --dataset "lttm" --root_path "D:/Datasets/CarlaLTTM" --splits_path "D:/Datasets/CarlaLTTM/splits" --train_split "train_mc" --val_split "val_mc" --time_of_day "noon" --logdir "logs/carla_msiw" --classifier "DeepLabV2MSIW" --target_dataset "city" --target_root_path "F:/Dataset/Cityscapes_extra" --target_splits_path "F:/Dataset/Cityscapes_extra" --iterations 25000 --ckpt_file "C:/Users/barbato/Desktop/carla_analyses/source_only/carla_mc/latest.pth"

python tools/train_UDA.py --dataset "gta" --root_path "F:/Dataset/GTA/full" --splits_path "F:/Dataset/GTA/full" --logdir "logs/gta_msiw" --classifier "DeepLabV2MSIW" --target_dataset "city" --target_root_path "F:/Dataset/Cityscapes_extra" --target_splits_path "F:/Dataset/Cityscapes_extra" --iterations 25000 --ckpt_file "C:/Users/barbato/Desktop/carla_analyses/source_only/gta/latest.pth"

python tools/train_UDA.py --dataset "idda" --root_path "D:/Datasets/IDDAbest" --splits_path "D:/Datasets/IDDAbest" --logdir "logs/idda_msiw" --classifier "DeepLabV2MSIW" --target_dataset "city" --target_root_path "F:/Dataset/Cityscapes_extra" --target_splits_path "F:/Dataset/Cityscapes_extra" --class_set "idda16" --iterations 25000 --ckpt_file "C:/Users/barbato/Desktop/carla_analyses/source_only/idda/latest.pth"