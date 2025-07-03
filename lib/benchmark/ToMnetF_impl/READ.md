## Directory Structure

The implementation uses experiment-specific directories for better organization and extensibility:

```
ToMnetF_impl/
├── scripts
│   ├── /{experiment_type}        # Implemented code by experiment
│   │   ├── tomnet.py             # ToMnet architecture
│   │   ├── agents.py             # RandomAgent and GoalDirectedAgent
│   │   ├── train.py              # Advanced training system
│   │   ├── evaluate.py           # Cross-species evaluation and metrics
│   │   └── visualize.py          # Publication-quality visualization
│   ├── data_generation.py        # Trajectory collection and batch formation (RunAgent.py)
│   └── environment.py            # LabMaze GridWorld environment
├── shell/                        # Automation scripts
│   └── run_exp3.sh              # Complete workflow automation (train - evaluate - visualize)
├── data/{experiment_type}/       # Training data organized by experiment
│   ├── alpha_0.01.pkl
│   ├── alpha_0.03.pkl
│   └── ...
├── models/{experiment_type}/     # Trained models organized by experiment
│   ├── 0.01_best.pth
│   └── 0.03_best.pth
├── result/{experiment_type}/     # Results organized by experiment
│   ├── training_results.json
│   ├── evaluation_results.pkl
│   ├── model_paths.json
│   ├── data_paths.json
│   └── run_cross_species_evaluation.sh
├── plots/{experiment_type}/      # Generated plots organized by experiment
│   ├── a_action_likelihood.png
│   ├── b_character_embeddings.png
│   ├── c_cross_species_kl.png
│   └── d_mixed_species.png
└── log/                         # Execution logs with timestamps
    ├── training/{timestamp}/
    ├── evaluation/{timestamp}/
    └── visualization/{timestamp}/
```