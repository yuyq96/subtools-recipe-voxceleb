# D-TDNN recipe for ASV-Subtools

⚠️ This project is currently under development.

⚠️ The training settings in this recipe and those in the paper are different.


```
cd $kaldi_root/egs
mkdir asv && cd asv
git clone https://github.com/yuyq96/asv-subtools subtools
cd subtools/recipe
git clone https://github.com/yuyq96/subtools-recipe-voxceleb-dtdnn voxceleb-dtdnn
cd ../..
subtools/recipe/runVoxceleb.sh
```
