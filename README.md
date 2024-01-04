# Masterarbeit

This repo contains the code for my master thesis "The Impact of Limited Predictions from Multiple Experts on Learning to Defer Systems"

Some of the code is from
- https://github.com/ptrckhmmr/learning-to-defer-with-limited-expert-predictions
    - In the SSL folder
- https://github.com/ptrckhmmr/human-ai-teams
    - In the Hemmer folder
- https://github.com/rajevv/Multi_L2D
    - In the Verma folder
- https://github.com/clinicalml/active_learn_to_defer
    - In the AL folder

The full experiment is located in the experiment.py file


## How to run

The experiment file gets the save path for the metrics, the number of workers and the experiment parameters as input like this

    python experiment.py /path/for/metrics/to/save  number_of_workers  experiment_file.json

The configuration for the single experiments is in:
- NIH_Experiment.json for the NIH Dataset
- CIFAR100_Experiment.json for the CIFAR-100 dataset

## Experiment adjustments
The experiment configuration allows to run different parameter combinations
- Setting: ["AL", "SSL", "SSL_AL", "SSL_AL_SSL", "PERFECT"]
	- AL -> Active Learning (Supervised)
	- SSL -> Semi-Supervised Learning
	- SSL_AL -> Semi-Supervised-Learning with Active Learning (supervised)
	- SSL_AL_SSL -> Semi-Supervised-Learning with Active Learning (semi-supervised)
	- PERFECT -> Fully labeled setting
- Mod: ["confidence", "disagreement_diff", "ssl", "perfect"] -> Active Learning strategy
	- Confidence -> Entropy on all unlabeled data
	- Disagreement_diff -> Entropy on the disagreement set
	- ssl, perfect -> for compatibility for the ssl, perfect setting
- Overlap [0, 100] -> Percentage of same images labeled at the beginning
- Sample_Equal [True, False] -> If the target class is evenly distributed in the starting images
- Expert_Predict ["right", "target"] -> How the human is modeled
	- right -> The expert model predicts if the human is right
	- target -> The expert model mimics the human
- Initial_Size [int] -> Number of images labeled at the beginning
- Rounds [int] -> Number of active learning rounds
- Labels per round [int] -> Number of images selected in each AL round
- COST [(int, int)] -> Cost for (right, wrong) if the expert model predicts if the expert is right
	- Can be used to balance the classes
