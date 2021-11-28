setlocal
set ckptpath=Y:/members/barbato/Code/CV_Analyses/logs/city_PSP/
set prefix=city_psp
set class=PSPNet

python tools/test.py --dataset "city" --root_path "F:/Dataset/Cityscapes_extra" --splits_path "F:/Dataset/Cityscapes_extra" --logdir "logs/test/%prefix%" --rescale_size "1280," --classifier "%class%" --ckpt_file "%ckptpath%val_best.pth"

python tools/test.py --dataset "mapi" --root_path "F:/Dataset/Mapillary" --splits_path "F:/Dataset/Mapillary" --logdir "logs/test/%prefix%_to_mapi" --rescale_size "1280," --crop_images True --crop_size "1280,640" --classifier "%class%" --ckpt_file "%ckptpath%val_best.pth"

python tools/test.py --dataset "idd" --root_path "F:/Dataset/IDD" --splits_path "F:/Dataset/IDD" --logdir "logs/test/%prefix%_to_idd" --rescale_size "1280," --classifier "%class%" --ckpt_file "%ckptpath%val_best.pth"

python tools/test.py --dataset "lttm" --root_path "D:/Datasets/CarlaLTTM" --splits_path "D:/Datasets/CarlaLTTM" --weather "clear" --time_of_day "noon" --logdir "logs/test/%prefix%_to_ccn" --rescale_size "1280," --classifier "%class%" --ckpt_file "%ckptpath%val_best.pth"

python tools/test.py --dataset "gta" --root_path "F:/Dataset/GTA/full" --splits_path "F:/Dataset/GTA/full" --logdir "logs/test/%prefix%_to_gta" --rescale_size "1280," --classifier "%class%" --ckpt_file "%ckptpath%val_best.pth" --test_split "val"