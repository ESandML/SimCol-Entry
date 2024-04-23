import sys
import os
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models


def train_epoch(model, device, train_loader, optimizer, epoch, loss):

    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_out = loss(output, target)
        loss_out.backward()
        optimizer.step()
        loss_accumulator.append(loss_out.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss_out.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator)


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf = perf_measure(output, target).item()
        for i in range(len(output)):
            perf_accumulator.append(perf)
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    len(perf_accumulator),
                    len(test_loader.dataset),
                    100.0 * len(perf_accumulator) / len(test_loader.dataset),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    len(perf_accumulator),
                    len(test_loader.dataset),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(perf_accumulator), np.std(perf_accumulator)


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    base_folders = sorted(glob.glob(args.root + "/*/"))
    sub_folders = []
    for bf in base_folders:
        sub_folders += sorted(glob.glob(bf + "*/"))

    val_vids = [
        args.root + "/SyntheticColon_I/Frames_S4/",
        args.root + "/SyntheticColon_I/Frames_S9/",
        args.root + "/SyntheticColon_I/Frames_S14/",
        args.root + "/SyntheticColon_II/Frames_B4/",
        args.root + "/SyntheticColon_II/Frames_B9/",
        args.root + "/SyntheticColon_II/Frames_B14/",
    ]
    test_vids = [
        args.root + "/SyntheticColon_I/Frames_S5/",
        args.root + "/SyntheticColon_I/Frames_S10/",
        args.root + "/SyntheticColon_I/Frames_S15/",
        args.root + "/SyntheticColon_II/Frames_B5/",
        args.root + "/SyntheticColon_II/Frames_B10/",
        args.root + "/SyntheticColon_II/Frames_B15/",
        args.root + "/SyntheticColon_III/Frames_O1/",
        args.root + "/SyntheticColon_III/Frames_O2/",
        args.root + "/SyntheticColon_III/Frames_O3/",
    ]
    train_vids = sub_folders
    for vid in test_vids + val_vids:
        train_vids.remove(vid)

    train_depth = []
    train_rgb = []
    val_depth = []
    val_rgb = []

    for vid in train_vids:
        files = sorted(glob.glob(vid + "*.png"))
        train_depth += [f for f in files if f.split("/")[-1].startswith("Depth")]
        train_rgb += [f for f in files if f.split("/")[-1].startswith("Frame")]
    assert len(train_depth) == len(train_rgb)
    for vid in val_vids:
        files = sorted(glob.glob(vid + "*.png"))
        val_depth += [f for f in files if f.split("/")[-1].startswith("Depth")]
        val_rgb += [f for f in files if f.split("/")[-1].startswith("Frame")]
    assert len(val_depth) == len(val_rgb)

    # Remove bad frames
    if args.root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0059.png" in val_rgb:
        val_rgb.remove(args.root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0059.png")
    if args.root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0060.png" in val_rgb:
        val_rgb.remove(args.root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0060.png")
    if args.root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0061.png" in val_rgb:
        val_rgb.remove(args.root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0061.png")
    if args.root + "/SyntheticColon_I/Frames_S14/Depth_0059.png" in val_depth:
        val_depth.remove(args.root + "/SyntheticColon_I/Frames_S14/Depth_0059.png")
    if args.root + "/SyntheticColon_I/Frames_S14/Depth_0060.png" in val_depth:
        val_depth.remove(args.root + "/SyntheticColon_I/Frames_S14/Depth_0060.png")
    if args.root + "/SyntheticColon_I/Frames_S14/Depth_0061.png" in val_depth:
        val_depth.remove(args.root + "/SyntheticColon_I/Frames_S14/Depth_0061.png")

    train_dataloader, val_dataloader = dataloaders.get_dataloaders(
        train_depth, train_rgb, val_depth, val_rgb, batch_size=args.batch_size
    )

    loss = nn.MSELoss()
    if args.architecture == "FCBFormer_D":
        model = models.FCBFormer_D()
    elif args.architecture == "UNet":
        model = models.UNet()

    if args.cont == "true":
        state_dict = torch.load(
            "Trained models/{}_weights.pt".format(args.architecture), map_location="cpu"
        )
        model.load_state_dict(state_dict["model_state_dict"])

    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.cont == "true":
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        start_epoch = state_dict["epoch"] + 1
        prev_best_test = state_dict["test_measure_mean"]
    else:
        start_epoch = 1
        prev_best_test = None

    return (
        device,
        train_dataloader,
        val_dataloader,
        loss,
        model,
        optimizer,
        start_epoch,
        prev_best_test,
    )


def train(args):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    (
        device,
        train_dataloader,
        val_dataloader,
        loss,
        model,
        optimizer,
        start_epoch,
        prev_best_test,
    ) = build(args)

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, verbose=True
            )
    for epoch in range(start_epoch, args.epochs + 1):
        try:
            av_loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, loss
            )
            test_measure_mean, test_measure_std = test(
                model, device, val_dataloader, epoch, loss
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            scheduler.step(test_measure_mean)
        if prev_best_test == None or test_measure_mean < prev_best_test:
            print("Saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": av_loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                "Trained models/{}_weights.pt".format(args.architecture),
            )
            prev_best_test = test_measure_mean


def get_args():
    parser = argparse.ArgumentParser(description="Train model on SimCol depth")
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--architecture", type=str, default="FCBFormer_D")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=0.0, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="true", dest="mgpu", choices=["true", "false"]
    )
    parser.add_argument("--continue", type=str, default="false", dest="cont")

    return parser.parse_args()


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()

