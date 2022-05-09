# fetch pre-trained teacher models

mkdir -p save/models/

cd save/models

mkdir -p wrn_40_2_vanilla
wget http://shape2prog.csail.mit.edu/repo/wrn_40_2_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth wrn_40_2_vanilla/

mkdir -p resnet32x4_vanilla
wget http://shape2prog.csail.mit.edu/repo/resnet32x4_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth resnet32x4_vanilla/

cd ../..