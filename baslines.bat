python tools/train.py --dataset "mapi" --root_path "F:/Dataset/Mapillary" --splits_path "F:/Dataset/Mapillary" --logdir "logs/mapi_FCN" --rescale_size "1280," --crop_images True --crop_size "1280,640" --classifier "FCN"

python tools/train.py --dataset "idd" --root_path "F:/Dataset/IDD" --splits_path "F:/Dataset/IDD" --logdir "logs/idd_FCN" --rescale_size "1280," --classifier "FCN"

python tools/train.py --dataset "lttm" --root_path "D:/Datasets/CarlaLTTM" --splits_path "D:/Datasets/CarlaLTTM" --weather "clear" --time_of_day "noon" --logdir "logs/carla_clearnoon_FCN" --rescale_size "1280," --classifier "FCN"

python tools/train.py --dataset "city" --root_path "F:/Dataset/Cityscapes_extra" --splits_path "F:/Dataset/Cityscapes_extra" --logdir "logs/city_FCN" --rescale_size "1280," --classifier "FCN"

python tools/train.py --dataset "gta" --root_path "F:/Dataset/GTA/full" --splits_path "F:/Dataset/GTA/full" --logdir "logs/gta_FCN" --rescale_size "1280," --classifier "FCN"