# srd
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --distill srd -a 0 -b 1 --ood tin --v2
# kd
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth--model_s ShuffleV1 --distill kd -r 0.1 -a 0.9 -b 0 --ood tin --v2
# crd
#python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  --model_s ShuffleV1  --distill crd -a 0 -b 0.8 --ood tin --v2
# FitNet
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth   --model_s ShuffleV1 --distill hint -a 0 -b 100 --ood tin --v2
# AT
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  --model_s ShuffleV1 --distill attention -a 0 -b 1000 --ood tin --v2
# SP
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --distill similarity  -a 0 -b 3000 --ood tin --v2
# CC
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  --model_s ShuffleV1 --distill correlation -a 0 -b 0.02 --ood tin --v2
# VID
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  --model_s ShuffleV1 --distill vid -a 0 -b 1 --ood tin --v2
# RKD
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --distill rkd  -a 0 -b 1 --ood tin --v2
# PKT
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --distill pkt  -a 0 -b 30000 --ood tin --v2
# AB
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  --model_s ShuffleV1 --distill abound -a 0 -b 1 --ood tin --v2
# FT
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  --model_s ShuffleV1 --distill factor -a 0 -b 200 --ood tin --v2
# FSP
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --distill fsp  -a 0 -b 50 --ood tin --v2
# NST
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --distill nst  -a 0 -b 50 --ood tin --v2
