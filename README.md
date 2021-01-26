# D-TDNN recipe for ASV-Subtools

⚠️ This project is currently under development.

⚠️ The training settings are different from those in the paper:

- ~~batch size (128)~~
- learning rate scheduler
- ~~weight decay (5e-4)~~

|  | Recipe | Paper |
| :- | :-: | :-: |
| Train on | VoxCeleb2 dev + aug * 4 | VoxCeleb1 dev + VoxCeleb2 |
| VAD | ✔️ | ✖️ |
| # Feature | 81 | 23 |
| # Frame | 200 | 200-400 |
| Nonlinearity | ReLU | PReLU |
| Margin | 0.25 | 0.4 |
| Scaling factor | 32 | 64 |

## Usage

```
cd $kaldi_root/egs
mkdir asv && cd asv
git clone https://github.com/yuyq96/asv-subtools subtools
cd subtools/recipe
git clone https://github.com/yuyq96/subtools-recipe-voxceleb-dtdnn voxceleb-dtdnn
cd ../..
subtools/recipe/runVoxceleb.sh
```
