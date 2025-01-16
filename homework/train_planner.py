import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tensorboard

from homework.metrics import PlannerMetric

from .models import load_model, save_model
from .datasets.road_dataset import load_data


def train(
    log_directory: str = "logs",
    epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    random_seed: int = 2024,
    **kwargs,
):
    planner_model = "cnn_planner"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_path = Path(log_directory) / f"{planner_model}_{datetime.now().strftime('%m%d_%H%M%S')}"
    writer = tensorboard.SummaryWriter(log_path)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(planner_model, **kwargs)
    model = model.to(device)
    model.train()

    training_loader = load_data(
        "drive_data/train",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
        transform_pipeline="default"
    )
    validation_loader = load_data("drive_data/val", shuffle=False)

    # create loss function and optimizer
    criterion = torch.nn.L1Loss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    step = 0
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()
    loss_history = {
        "train_loss": [],
    }

    # training loop
    for epoch in range(epochs):
        # clear metrics at beginning of epoch
        train_metric.reset()
        val_metric.reset()

        model.train()

        for batch in validation_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            images = batch["image"]
            targets = batch["waypoints"]
            masks = batch["waypoints_mask"]

            outputs = model(images)
            train_metric.add(outputs, targets, masks)

            loss_values = criterion(outputs, targets)
            masked_loss = loss_values * masks[..., None]
            loss = masked_loss.sum() / masks.sum()
            
            loss_history["train_loss"].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in training_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                images = batch["image"]
                targets = batch["waypoints"]
                masks = batch["waypoints_mask"]

                outputs = model(images)
                val_metric.add(outputs, targets, masks)

        # log average training loss
        writer.add_scalar("train_loss", torch.as_tensor(loss_history["train_loss"]).mean(), step)

        train_results = train_metric.compute()
        val_results = val_metric.compute()

        train_l1_error = torch.as_tensor(train_results["l1_error"])
        train_long_error = torch.as_tensor(train_results["longitudinal_error"])
        train_lat_error = torch.as_tensor(train_results["lateral_error"])

        val_l1_error = torch.as_tensor(val_results["l1_error"])
        val_long_error = torch.as_tensor(val_results["longitudinal_error"])
        val_lat_error = torch.as_tensor(val_results["lateral_error"])

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Train L1 Error: {train_l1_error:.4f} "
                f"Val L1 Error: {val_l1_error:.4f} "
                f"Train Longitudinal Error: {train_long_error:.4f} "
                f"Val Longitudinal Error: {val_long_error:.4f} "
                f"Train Lateral Error: {train_lat_error:.4f} "
                f"Val Lateral Error: {val_lat_error:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_path / f"{planner_model}.th")
    print(f"Model saved to {log_path / f'{planner_model}.th'}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--log_directory", type=str, default="logs")
    arg_parser.add_argument("--planner_model", type=str, required=True)
    arg_parser.add_argument("--epochs", type=int, default=50)
    arg_parser.add_argument("--learning_rate", type=float, default=1e-3)
    arg_parser.add_argument("--random_seed", type=int, default=2024)

    # optional: additional model hyperparameters
    # arg_parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(arg_parser.parse_args()))