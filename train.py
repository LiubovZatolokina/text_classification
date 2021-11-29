import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import load_dataset
from model import AttentionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
num_epochs = 300
model_saving_path = './lstm_attention.pt'


def train_model(model, dataloaders, criterion, optimizer, num_epochs, model_saving_path):
    since = time.time()
    tb = SummaryWriter()
    max_f1_score = -1
    for epoch in tqdm(range(num_epochs)):
        loss_dict = {}
        acc_dict = {}
        f1score_dict = {}
        for phase in ['train', 'valid']:
            preds_list = []
            labels_lit = []
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for idx, batch in enumerate(dataloaders[phase]):
                inputs = batch.text.to(device)
                labels = batch.label.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                preds_list.append(preds.cpu().numpy())
                labels_lit.append(labels.data.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_f1score = f1_score(np.concatenate(preds_list), np.concatenate(labels_lit), average='weighted')
            print('Loss', epoch_loss)
            print('Accuracy', epoch_acc)
            print('F1 score', epoch_f1score)
            loss_dict[phase] = epoch_loss
            acc_dict[phase] = epoch_acc
            f1score_dict[phase] = epoch_f1score
            if epoch_f1score > max_f1_score:
                torch.save(model.state_dict(), model_saving_path)
                max_f1_score = epoch_f1score

        tb.add_scalars('Loss: epoch', {'Train': loss_dict['train'], 'Valid': loss_dict['valid']}, epoch)
        tb.add_scalars('Accuracy: epoch', {'Train': acc_dict['train'], 'Valid': acc_dict['valid']}, epoch)
        tb.add_scalars('F1 score: epoch', {'Train': f1score_dict['train'], 'Valid': f1score_dict['valid']}, epoch)



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    TEXT, LABEL, train_loader, test_loader = load_dataset()

    input_size = len(TEXT.vocab)
    output_size = len(LABEL.vocab)
    hidden_size = 512
    emb_size = 256

    model = AttentionModel(output_size, hidden_size, input_size, emb_size)
    torch.cuda.empty_cache()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataloaders_dict = {'train': train_loader, 'valid': test_loader}

    model_ft = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, model_saving_path)
