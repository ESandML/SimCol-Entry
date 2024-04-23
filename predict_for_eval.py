import torch
import os
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_vids = [
        args.root + "/SyntheticColon_I/Frames_S5/*.png",
        args.root + "/SyntheticColon_I/Frames_S10/*.png",
        args.root + "/SyntheticColon_I/Frames_S15/*.png",
        args.root + "/SyntheticColon_II/Frames_B5/*.png",
        args.root + "/SyntheticColon_II/Frames_B10/*.png",
        args.root + "/SyntheticColon_II/Frames_B15/*.png",
        args.root + "/SyntheticColon_III/Frames_O1/*.png",
        args.root + "/SyntheticColon_III/Frames_O2/*.png",
        args.root + "/SyntheticColon_III/Frames_O3/*.png",
    ]
    test_rgb = []

    for vid in test_vids:
        if not os.path.exists(os.path.join(os.path.dirname(vid) + "_OP", "depth")):
            os.makedirs(os.path.join(os.path.dirname(vid) + "_OP", "depth"))
        files = sorted(glob.glob(vid))
        test_rgb += [f for f in files if f.split("/")[-1].startswith("Frame")]

    OP_files = []
    for filename in test_rgb:
        OP_files.append(
            os.path.join(
                os.path.dirname(filename) + "_OP",
                "depth",
                os.path.basename(filename).replace(".png", ".npy"),
            )
        )

    dataloader = dataloaders.get_dataloaders_test(test_rgb)

    if args.architecture == "FCBFormer_D":
        model = models.FCBFormer_D()
    elif args.architecture == "UNet":
        model = models.UNet()

    state_dict = torch.load(
        f"./Trained models/{args.architecture}_weights.pt", map_location="cpu"
    )
    model.load_state_dict(state_dict["model_state_dict"])

    model.to(device)

    return device, dataloader, model, OP_files


@torch.no_grad()
def predict(args):
    device, test_dataloader, model, OP_files = build(args)

    t = time.time()
    model.eval()
    for i, data in enumerate(test_dataloader):
        data = data.to(device)
        output = model(data)
        predicted_map = output.cpu().numpy()
        predicted_map = np.squeeze(predicted_map)
        np.save(
            OP_files[i],
            np.float16(predicted_map),
        )
        if i + 1 < len(test_dataloader):
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    time.time() - t,
                )
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions for final evaluation"
    )
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--architecture", type=str, default="FCBFormer_D")

    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()

