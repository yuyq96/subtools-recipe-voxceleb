# D-TDNN recipe for ASV-Subtools

⚠️ This project is currently under development.

⚠️ The training settings are different from those in the paper:

- [ ] LR scheduler
- [x] ~~Batch size (128)~~
- [x] ~~Weight decay (5e-4)~~

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
| Momentum (SGD) | 0.9 | 0.95 |

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
