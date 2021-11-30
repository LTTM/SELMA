setlocal
set ckptpath=C:/Users/barbato/Desktop/
set prefix=gta
set class=DeepLabV3

python tools/test.py --dataset "city" --root_path "F:/Dataset/Cityscapes_extra" --splits_path "F:/Dataset/Cityscapes_extra" --logdir "logs/test/%prefix%/%prefix%_to_city" --rescale_size "1280," --classifier "%class%" --ckpt_file "%ckptpath%val_best.pth"

python tools/test.py --dataset "mapi" --root_path "F:/Dataset/Mapillary" --splits_path "F:/Dataset/Mapillary" --logdir "logs/test/%prefix%/%prefix%_to_mapi" --rescale_size "1280," --crop_images True --crop_size "1280,640" --classifier "%class%" --ckpt_file "%ckptpath%val_best.pth"

python tools/test.py --dataset "idd" --root_path "F:/Dataset/IDD" --splits_path "F:/Dataset/IDD" --logdir "logs/test/%prefix%/%prefix%_to_idd" --rescale_size "1280," --classifier "%class%" --ckpt_file "%ckptpath%val_best.pth"

python tools/test.py --dataset "lttm" --root_path "D:/Datasets/CarlaLTTM" --splits_path "D:/Datasets/CarlaLTTM" --weather "clear" --time_of_day "noon" --logdir "logs/test/%prefix%/%prefix%_to_ccn" --rescale_size "1280," --classifier "%class%" --ckpt_file "%ckptpath%val_best.pth"

python tools/test.py --dataset "gta" --root_path "F:/Dataset/GTA/full" --splits_path "F:/Dataset/GTA/full" --logdir "logs/test/%prefix%/%prefix%_to_gta" --rescale_size "1280," --classifier "%class%" --ckpt_file "%ckptpath%val_best.pth" --test_split "val"


REM python tools/test.py --dataset "city" --root_path "F:/Dataset/Cityscapes_extra" --splits_path "F:/Dataset/Cityscapes_extra" --logdir "logs/test/carla_cn/carla_cn_to_city_dlv2" --rescale_size "1280," --classifier "DeepLabV2" --ckpt_file "C:/Users/barbato/Desktop/val_best.pth"

REM python tools/test.py --dataset "city" --root_path "F:/Dataset/Cityscapes_extra" --splits_path "F:/Dataset/Cityscapes_extra" --logdir "logs/test/carla_mcnoon_DLV2b/city_dlv2_e3_noise50" --rescale_size "1280," --classifier "DeepLabV2" --ckpt_file "logs/carla_mcnoon_DLV2b_sn/latest.pth"

REM python tools/test.py --dataset "lttm" --root_path "D:/Datasets/CarlaLTTM" --splits_path "D:/Datasets/CarlaLTTM/splits" --train_split "train_mc" --val_split "val_mc" --test_split "test_mc" --time_of_day "noon" --logdir "logs/test/carla_mcnoon/carla_mcnoon_DLV2b" --rescale_size "1280," --classifier "DeepLabV2MSIW" --ckpt_file "logs/carla_mcnoon_DLV2b/val_best.pth"