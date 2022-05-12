python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth  --model_s wrn_40_1 --distill abound -a 0 -b 1  --ood tin
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --distill abound -a 0 -b 1 --ood tin
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --distill abound -a 0 -b 1  --ood tin
