import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from BERT.Dataset import BertDataset
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from tqdm import tqdm


def trainer(group_column,
            data_column,
            model,
            data,
            optimizer = None,
            epochs = 10,
            output_dir = "./models",
            train_history = 128,
            validation = 5):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    losses = []
    accuracies = []

    optimizer = optimizer(model.parameters(), lr = 1e-5)

    train_data, test_data = train_test_split(data, test_size = 0.2, random_state = 42)

    train_dataset = BertDataset(train_data, group_column, data_column, train_history=train_history)

    train_loader = DataLoader(train_dataset, batch_size = 32, num_workers = 1)

    for i in range(1, epochs + 1):
        print(f"Epoch {i}")
        train_loss, train_acc = train_step(model, "cpu", train_loader, optimizer)
        losses += [train_loss]
        accuracies += [train_acc]
        print(f"Train loss: {train_loss}, Train accuracy: {train_acc}")


def train_step(model,
               device,
               loader,
               optimizer,
               MASK_TOKEN = 1):

    model.train()
    total_loss = 0
    total_counts = 0
    train_accs = []
    train_bs = []

    print("loader:", loader)

    for _, batch in enumerate(tqdm(loader)):

        source = Variable(batch["source"]).to(device)
        target = Variable(batch["target"]).to(device)
        source_mask = Variable(batch["source_mask"]).to(device)
        target_mask = Variable(batch["target_mask"]).to(device)

        train_bs += [source.size(0)]

        mask = target_mask == MASK_TOKEN

        optimizer.zero_grad()

        output = model(source, source_mask)

        loss = calculate_loss(output, target, mask)

        total_counts += 1
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(output, target, mask)
        train_accs += [acc.item()]

    epoch_mean = calculate_combined_mean(train_bs, train_accs)

    return (total_loss / total_counts) if total_counts else 0, sum(train_accs)/len(train_accs)


def calculate_loss(y_pred, y_true, mask):

    # y_pred = y_pred.view(-1, y_pred.size(2))
    # y_true = y_true.view(-1)
    loss = F.cross_entropy(y_pred, y_true, reduction="none", ignore_index=0)
    # return loss
    loss = loss * mask
    loss = loss.sum() / (mask.sum() + 1e-8)
    return loss


def calculate_accuracy(y_pred, y_true, mask):

    _, prediction = y_pred.max(1)
    y_true = torch.masked_select(y_true, mask)
    prediction = torch.masked_select(prediction, mask)

    return (y_true == prediction).double().mean()


def calculate_combined_mean(batch_sizes, means):

    combined_mean = (torch.sum(torch.tensor(batch_sizes) * torch.tensor(means)) /
                     torch.sum(torch.tensor(batch_sizes))) * 100
    return combined_mean