# RoadSense: Robust Traffic Sign Recognition (Group 4)

Atria University, Neural Networks and CNN Sprint. A CNN that classifies the 43 GTSRB sign classes,
stress-tested with corrupted images (noise, blur, brightness, contrast, occlusion, JPEG).

## Setup
```bash
pip install -r requirements.txt
```
**Dataset (not in the repo):** download the Kaggle mirror
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
In a Kaggle Notebook use "Add Data" and the path is `/kaggle/input/gtsrb-german-traffic-sign`.

## How to run
```bash
python -m src.train --data-root <path-to-gtsrb> --epochs 15 --name baseline
```
Outputs go to `results/` (metrics JSON, confusion pairs, robustness CSV) and `results/figures/`.

## Key design choices
- **Leakage-free split:** train/val split by *track* (`GroupShuffleSplit` on class + track), official test set kept untouched.
- **Label-safe augmentation:** no horizontal flips (keep left <-> keep right).
- **Robustness bench:** `src/corruptions.py`, 7 corruptions x 3 severities, fixed seed.
- Seed = 42 everywhere.

## Results
| Model | Clean acc | Macro-F1 | Mean corruption acc |
|---|---|---|---|
| Baseline CNN | TBD | TBD | TBD |

## Contribution table
| Member | GitHub username | Responsibilities | Key files |
|---|---|---|---|
| Aseem (Team Leader) | TBD | Augmentation, transfer learning, coordination | TBD |
| Sohail | TBD | Data, EDA, split | `notebooks/01_eda.ipynb`, `src/dataset.py` |
| Raju | TBD | Scratch CNN, error analysis | `src/models.py`, `src/train.py` |
| Manoj | TBD | Robustness bench | `src/corruptions.py`, `src/evaluate.py` |

## AI assistance
Starter code was drafted with an AI assistant (Claude) and reviewed by the team. (Update with exact details.)
