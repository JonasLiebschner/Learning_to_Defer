import os
import json
import argparse

from src.CIFAR100_Expert import CIFAR100_Expert
import src.preprocess_data as prep


def generate_synthetic_expert(strength=60, binary=False, num_classes=20, per_s=1.0, per_w=0.0, seed=42, path="./", name=None):

    args = {
        "strength": strength,
        "binary": binary,
        "num_classes": num_classes,
        "per_s": per_s,
        "per_w": per_w
    }

    #Create unique seed for this expert
    path_name = expert_to_path(name, args, seed)

    # generate expert of strength X
    expert = CIFAR100_Expert(args.num_classes, args.strength, args.per_s, args.per_w, seed=path_name, name=name)

    train_data, test_data = prep.get_train_test_data()

    true_ex_labels = {'train': expert.generate_expert_labels(train_data.targets, binary=args.binary).tolist(),
                      'test': expert.generate_expert_labels(test_data.targets, binary=args.binary).tolist()}

    os.makedirs(f'{path}/synthetic_experts/{path_name}', exist_ok=True)
    with open(f'{path}/synthetic_experts/{path_name}/cifar100_expert_{args.strength}_labels.json', 'w') as f:
        json.dump(true_ex_labels, f)

    args["name"] = name
    with open(f'{path}synthetic_experts/{path_name}/cifar100_expert_params.json', 'w') as f:
        json.dump(args, f)

    return true_ex_labels

def expert_to_path(name, args, seed):
    args = args.copy()
    args["name"] = name
    args["seed"] = seed
    return param_to_path(args)

def param_to_path(args):
    s = ""
    for key, value in args.items():
        s += f"{key}_{value}_"
    s = s[:-1]
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8