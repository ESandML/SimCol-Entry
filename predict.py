import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.testorval == "test":
        vids = [
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
    elif args.testorval == "val":
        vids = [
            args.root + "/SyntheticColon_I/Frames_S4/",
            args.root + "/SyntheticColon_I/Frames_S9/",
            args.root + "/SyntheticColon_I/Frames_S14/",
            args.root + "/SyntheticColon_II/Frames_B4/",
            args.root + "/SyntheticColon_II/Frames_B9/",
            args.root + "/SyntheticColon_II/Frames_B14/",
        ]

    rgb = []
    for vid in vids:
        files = sorted(glob.glob(vid))
        rgb += [f for f in files if f.split("/")[-1].startswith("Frame")]

    dataloader = dataloaders.get_dataloaders_test(rgb)

    if args.architecture == "FCBFormer_D":
        model = models.FCBFormer_D()
    elif args.architecture == "UNet":
        model = models.UNet()

    state_dict = torch.load("./Trained models/{}_weights.pt".format(args.architecture))
    model.load_state_dict(state_dict["model_state_dict"])

    model.to(device)

    return device, dataloader, model


@torch.no_grad()
def predict(args):
    device, test_dataloader, model = build(args)

    if not os.path.exists(
        "./Predictions_{}/{}".format(args.architecture, args.testorval)
    ):
        os.makedirs("./Predictions_{}/{}".format(args.architecture, args.testorval))

    t = time.time()
    model.eval()
    perf_accumulator = []
    for i, data in enumerate(test_dataloader):
        data = data.to(device)
        output = model(data)
        predicted_map = output.cpu().numpy()
        predicted_map = np.squeeze(predicted_map)
        np.save(
            "./Predictions_{}/{}/{:04d}.npy".format(
                args.architecture, args.testorval, i
            ),
            predicted_map,
        )
        predicted_map = (predicted_map * 255 * 256).astype("uint16")
        cv2.imwrite(
            "./Predictions_{}/{}/{:04d}.png".format(
                args.architecture, args.testorval, i
            ),
            predicted_map,
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
    parser = argparse.ArgumentParser(description="Make predictions")
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--architecture", type=str, default="FCBFormer_D")
    parser.add_argument("--testorval", type=str, default="test")

    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()

