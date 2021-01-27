# ASV-Subtools Recipe

⚠️ This project is currently under development.

⚠️ The training settings in this recipe are different from those in the D-TDNN paper:

|  | Recipe | Paper |
| :- | :-: | :-: |
| Train on | VoxCeleb2 dev + aug * 4 | VoxCeleb1 dev + VoxCeleb2 |
| VAD | ✔️ | ✖️ |
| # Feature | 81 | 23 |
| # Frame | 200 | 200-400 |
| Nonlinearity | ReLU | PReLU |
| Momentum (last BN) | 0.5 | 0.1 |
| Margin | 0.25 | 0.4 |
| Scaling factor | 32 | 64 |
| Batch size | 256 | 128 |
| Momentum (SGD) | 0.9 | 0.95 |
| Initial LR | 0.02 | 0.01 |
| LR scheduler | ReduceLROnPlateau | Fixed |

⚠️ Up to now the training settings in the D-TDNN paper achieve better performance.

## Usage

```
cd $kaldi_root/egs
mkdir asv && cd asv
git clone https://github.com/yuyq96/asv-subtools.git subtools
cd subtools/recipe
git clone https://github.com/yuyq96/subtools-recipe-voxceleb.git voxceleb-dtdnn
cd ../..
subtools/recipe/runVoxceleb.sh
```
