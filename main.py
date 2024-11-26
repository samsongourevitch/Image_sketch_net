import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import models

from model_factory import ModelFactory


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        metavar="OPT",
        help="optimizer to use (default: SGD)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--feature_extractor_path",
        default=None,
        help="model usinng to extract features",
    )
    parser.add_argument(
        "--load_models", 
        type=str, 
        nargs='+',  # Allows one or more arguments for this option
        help="List of model paths to process"
    )
    parser.add_argument(
        "--train_all",
        action="store_true",
        help="Train all epochs without validation",
    )
    args = parser.parse_args()
    return args


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    device: torch.device,
    args: argparse.ArgumentParser,
    feature_extractor = None,
) -> float:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        if args.model_name == "sketch_classifier":
            feature_extractor.eval()
            feature_extractor.to(device)
            with torch.no_grad():
                features = feature_extractor(data)
                features = features.view(features.size(0), -1)
            output = model(features)
        else:
            output = model(data)
        optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        train_loss += loss.data.item()
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
    train_loss /= len(train_loader.dataset)
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    return train_loss


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    args: argparse.ArgumentParser,
    feature_extractor = None,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # if args.model_name == "sketch_classifier":
        #     feature_extractor.eval()
        #     with torch.no_grad():
        #         features = feature_extractor(data)
        #         features = features.view(features.size(0), -1)
        #     output = model(features)
        with torch.no_grad():
            output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return validation_loss

# Function to perform TTA
def tta_inference(model, image, tta_transforms, device):
    """
    Perform TTA on a single image.
    
    Args:
    - model: Pretrained model.
    - image: Input PIL Image.
    - tta_transforms: List of augmentations to apply.
    - device: 'cuda' or 'cpu'.
    
    Returns:
    - Final averaged prediction for the image.
    """
    predictions = []
    with torch.no_grad():
        for transform in tta_transforms:
            augmented_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
            pred = model(augmented_image)
            predictions.append(pred)
    
    # Average predictions
    final_prediction = torch.mean(torch.stack(predictions), dim=0)
    return final_prediction


def main():
    """Default Main Function."""
    # options
    args = opts()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform
    model, data_transforms = ModelFactory(args.model_name, load_models=args.load_models).get_all()
    if args.model_name == "sketch_classifier":
        resnet = models.resnet50(pretrained=True)
        feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.to(device)
        feature_extractor.eval()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Data initialization and loading
    print("Loading data...")
    print("Train data directory: " + args.data + "/train_images")

    if args.model_name == "efficientnet_based":
        args.batch_size = max(1, args.batch_size // 2)
        
    if args.train_all:
        print("Training on all available data (train + val)")
        # Combine train and validation datasets
        train_data = datasets.ImageFolder(args.data + "/train_images", transform=data_transforms)
        val_data = datasets.ImageFolder(args.data + "/val_images", transform=data_transforms)
        combined_data = torch.utils.data.ConcatDataset([train_data, val_data])

        # Use combined data for training
        train_loader = torch.utils.data.DataLoader(
            combined_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        # Validation loader is not used in this case
        val_loader = None
    else:
        print("Using separate train and validation datasets")
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data + "/val_images", transform=data_transforms),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    # Setup optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.to(device)

    train_losses = []
    val_losses = []

    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(1, args.epochs + 1):
        # training loop
        train_loss = train(model, optimizer, train_loader, use_cuda, epoch, device, args)
        train_losses.append(train_loss)
        if not args.train_all:
            # validation loop
            val_loss = validation(model, val_loader, use_cuda, args)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                # save the best model for validation
                best_val_loss = val_loss
                best_model_file = args.experiment + "/" + args.model_name + "_best.pth"
                torch.save(model.state_dict(), best_model_file)
        # also save the model every epoch
        if args.train_all:
            model_file = args.experiment + "/" + args.model_name + str(epoch) + "train_all.pth"
        else:
            model_file = args.experiment + "/" + args.model_name + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        if not args.train_all:
            print(
                "Saved model to "
                + model_file
                + f". You can run `python evaluate.py --model_name {args.model_name} --model "
                + best_model_file
                + "` to generate the Kaggle formatted csv file\n"
            )
    # save the training and validation loss
    with open(args.experiment + "/" + args.model_name + "_train_losses.txt", "w") as f:
        for item in train_losses:
            f.write("%s\n" % item)

    if not args.train_all:
        with open(args.experiment + "/" + args.model_name + "_val_losses.txt", "w") as f:
            for item in val_losses:
                f.write("%s\n" % item)

    print("Saved train losses to " + args.experiment + "/" + args.model_name + "_train_losses.txt")
    print("Saved validation losses to " + args.experiment + "/" + args.model_name + "_val_losses.txt")

if __name__ == "__main__":
    main()
