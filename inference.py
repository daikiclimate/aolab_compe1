import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from addict import Dict

from dataset import return_data
# from evaluator import evaluator
from models import build_model


def get_arg():
    parser = argparse.ArgumentParser(description="image model for action segmentation")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    return config


def main():
    config = get_arg()
    device = config.device
    test_set = return_data.return_testloader()
    n_fold = 5
    models = []
    config.save_folder = os.path.join(config.save_folder, config.model)
    for fold_num in range(n_fold):
        model = build_model.build_model(config)
        model = model.to(device)
        weight = torch.load(f"{config.save_folder}/model_{str(fold_num)}.pth")
        model.load_state_dict(weight)
        model.eval()
        models.append(model)

    preds = []
    pbar = tqdm.tqdm(total=len(test_set))
    for data in test_set:
        data = data.to(device)
        with torch.no_grad():
            output = [m(data) for m in models]
        output = torch.cat(output)
        output = torch.mean(output, 0).unsqueeze(0)
        output = torch.argmax(output, axis=1)
        preds.extend(output.cpu().detach().numpy())
        pbar.update(1)

    submit = pd.DataFrame(columns=["image_id", "labels"])
    submit["image_id"] = list(range(500))
    submit["labels"] = preds
    submit.to_csv(f"submits/submition_1.csv", index=False)
    print(submit["labels"].value_counts())


def count_labels(labels):
    num_labels = np.zeros(11, dtype=np.int)
    for l in labels:
        num_labels[l] += 1
    print(num_labels)


if __name__ == "__main__":
    main()
