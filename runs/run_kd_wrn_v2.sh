# srd
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill srd -a 0 -b 1 --ood tin --v2
# kd
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --distill kd -r 0.1 -a 0.9 -b 0 --ood tin --v2
# crd
#python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth  --model_s wrn_40_1  --distill crd -a 0 -b 0.8 --ood tin
# FitNet
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill hint  --model_s wrn_40_1 -a 0 -b 100 --ood tin --v2
# AT
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill attention --model_s wrn_40_1 -a 0 -b 1000 --ood tin --v2
# SP
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill similarity --model_s wrn_40_1 -a 0 -b 3000 --ood tin --v2
# CC
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill correlation --model_s wrn_40_1 -a 0 -b 0.02 --ood tin --v2
# VID
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill vid --model_s wrn_40_1 -a 0 -b 1 --ood tin --v2
# RKD
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill rkd --model_s wrn_40_1 -a 0 -b 1 --ood tin --v2
# PKT
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill pkt --model_s wrn_40_1 -a 0 -b 30000 --ood tin --v2
# AB
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill abound --model_s wrn_40_1 -a 0 -b 1 --ood tin --v2
# FT
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill factor --model_s wrn_40_1 -a 0 -b 200 --ood tin --v2
# FSP
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill fsp --model_s wrn_40_1 -a 0 -b 50 --ood tin --v2
# NST
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill nst --model_s wrn_40_1 -a 0 -b 50 --ood tin --v2
