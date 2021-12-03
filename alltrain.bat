python tools/train.py --dataset "idda" --root_path "D:/Datasets/IDDAbest" --splits_path "D:/Datasets/IDDAbest" --logdir "logs/idda_dlv2" --classifier "DeepLabV2MSIW" --target_dataset "city" --target_root_path "F:/Dataset/Cityscapes_extra" --target_splits_path "F:/Dataset/Cityscapes_extra" --validate_on_target True --class_set "idda16"

python tools/train.py --dataset "lttm" --root_path "D:/Datasets/CarlaLTTM" --splits_path "D:/Datasets/CarlaLTTM/splits" --train_split "train_mc" --val_split "val_mc" --test_split "test_mc" --time_of_day "noon" --logdir "logs/carla_dlv2" --classifier "DeepLabV2MSIW" --target_dataset "city" --target_root_path "F:/Dataset/Cityscapes_extra" --target_splits_path "F:/Dataset/Cityscapes_extra" --validate_on_target True

python tools/train.py --dataset "gta" --root_path "F:/Dataset/GTA/full" --splits_path "F:/Dataset/GTA/full" --logdir "logs/gta_dlv2" --classifier "DeepLabV2MSIW" --target_dataset "city" --target_root_path "F:/Dataset/Cityscapes_extra" --target_splits_path "F:/Dataset/Cityscapes_extra" --validate_on_target True