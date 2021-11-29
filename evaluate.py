import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc, RocCurveDisplay, \
    roc_curve, classification_report
from sklearn.metrics import f1_score

from model import AttentionModel, AttnModelManager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_accuracy(number_of_correct, data_amount):
    accuracy = number_of_correct / data_amount
    return accuracy


def calculate_f1_score(y_pred, y_true):
    f1score = f1_score(y_pred, y_true, average='weighted')
    return f1score


def create_classification_report(y_pred, y_true):
    return classification_report(y_pred, y_true, target_names=['0', '1'])


def create_confusion_matrix(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    disp_cm.plot()
    plt.savefig('images/confusion_matrix.png')


def create_roc_auc_curve(y_pred, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    disp_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                               estimator_name='example estimator')
    disp_roc.plot()
    plt.savefig('images/roc_auc.png')


def evaluate_model(model, eval_dataloader, criterion, optimizer):
    preds_list = []
    labels_lit = []
    model.eval()

    running_corrects = 0
    for idx, batch in enumerate(eval_dataloader):
        inputs = batch.text.to(device)
        labels = batch.label.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)
        preds_list.append(preds.cpu().numpy())
        labels_lit.append(labels.data.cpu().numpy())

    accuracy = calculate_accuracy(running_corrects.double(), len(eval_dataloader.dataset))

    f1score = calculate_f1_score(np.concatenate(preds_list), np.concatenate(labels_lit))

    print('Accuracy: ', accuracy)
    print('F1 score: ', f1score)
    print('Classification report: \n', create_classification_report(np.concatenate(preds_list),
                                                                  np.concatenate(labels_lit)))

    create_confusion_matrix(np.concatenate(labels_lit), np.concatenate(preds_list))

    create_roc_auc_curve(np.concatenate(labels_lit), np.concatenate(preds_list))


if __name__ == '__main__':
    am = AttnModelManager()
    model_ = AttentionModel(am.output_size, am.hidden_size, am.input_size, am.emb_size)
    model_.load_state_dict(torch.load(am.model_saving_path))
    model_ = model_.to(device)
    evaluate_model(model_, am.test_loader, am.criterion, am.optimizer)
