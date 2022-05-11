python train_pseudolabel.py --ood tin --arch wrn_40_1
python train_meanteacher.py --ood tin --arch wrn_40_1
python train_fixmatch.py --ood tin --arch wrn_40_1
python train_mixmatch.py --ood tin --arch wrn_40_1
python train_mtcr.py --ood tin --arch wrn_40_1
python train_openmatch.py --ood tin --arch wrn_40_1
python train_t2t_stage1.py --ood tin --arch wrn_40_1
python train_t2t_stage2.py --arch wrn_40_1 --gpu-id 0 --ood-dataset places --resume ./Results/T2T/Stage1/wrn_40_1_tin/checkpoint.pth.tar
