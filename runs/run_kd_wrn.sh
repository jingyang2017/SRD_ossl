# srd
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -a 0 -b 1 --ood tin
# kd
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill kd -r 0.1 -a 0.9 -b 0 --ood tin
# crd
#python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth  --model_s wrn_40_1  --distill crd -a 0 -b 0.8 --ood tin
# FitNet
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill hint  -a 0 -b 100 --ood tin
# AT
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill attention  -a 0 -b 1000 --ood tin
# SP
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill similarity  -a 0 -b 3000 --ood tin
# CC
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill correlation  -a 0 -b 0.02 --ood tin
# VID
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill vid  -a 0 -b 1 --ood tin
# RKD
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill rkd  -a 0 -b 1 --ood tin
# PKT
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill pkt  -a 0 -b 30000 --ood tin
# AB
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill abound  -a 0 -b 1 --ood tin
# FT
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill factor  -a 0 -b 200 --ood tin
# FSP
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill fsp  -a 0 -b 50 --ood tin
# NST
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill nst  -a 0 -b 50 --ood tin
