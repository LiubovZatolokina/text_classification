import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from dataset import load_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttentionModel(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, embedding_length):
        super(AttentionModel, self).__init__()

        self.hidden_size = hidden_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.lstm_1 = nn.LSTM(embedding_length, hidden_size)
        self.lstm_2 = nn.LSTM(embedding_length, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)
        self.linear_final = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_sentences):

        input_ = self.word_embeddings(input_sentences)
        input_ = input_.permute(1, 0, 2)
        query = [(torch.zeros(input_.shape[0], self.hidden_size),
                          torch.zeros(input_.shape[0], self.hidden_size)) for _ in range(2)]

        key, _ = self.lstm_1(input_)
        value, _ = self.lstm_2(input_)
        attn_output, _ = self.attention(query[1][0].unsqueeze(1).to(device), key.to(device), value.to(device))
        logits = self.linear_final(attn_output)

        return logits.squeeze(1)


class AttnModelManager():
    def __init__(self):
        super(AttnModelManager, self).__init__()
        self.TEXT, self.LABEL, self.train_loader, self.test_loader = load_dataset()
        self.input_size = len(self.TEXT.vocab)
        self.output_size = len(self.LABEL.vocab)
        self.hidden_size = 512
        self.emb_size = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attn_model = AttentionModel(self.output_size, self.hidden_size, self.input_size, self.emb_size)
        self.attn_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 300
        self.model_saving_path = './lstm_attention.pt'
        self.optimizer = torch.optim.Adam(self.attn_model.parameters(), lr=1e-4)
        self.dataloaders_dict = {'train': self.train_loader, 'valid': self.test_loader}

    def train(self):
        for epoch in range(self.num_epochs):
            preds_list = []
            labels_lit = []
            self.attn_model.train()

            running_loss = 0.0
            running_corrects = 0

            for idx, batch in enumerate(self.train_loader):
                inputs = batch.text.to(device)
                labels = batch.label.to(device)
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.attn_model(inputs)
                    loss = self.criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                preds_list.append(preds.cpu().numpy())
                labels_lit.append(labels.data.cpu().numpy())

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
            epoch_f1score = f1_score(np.concatenate(preds_list), np.concatenate(labels_lit), average='weighted')
            print('Loss', epoch_loss)
            print('Accuracy', epoch_acc)
            print('F1 score', epoch_f1score)

        torch.save(self.attn_model.state_dict(), self.model_saving_path)

        return self.attn_model

    def predict(self, input_):
        model_ = AttentionModel(self.output_size, self.hidden_size, self.input_size, self.emb_size)

        test_sen = self.TEXT.preprocess(input_)
        test_sen = [[self.TEXT.vocab.stoi[x] for x in test_sen]]
        test_sen = np.asarray(test_sen)

        model_.load_state_dict(torch.load(self.model_saving_path))
        model_ = model_.to(self.device)
        model_.eval()

        with torch.no_grad():
            output = model_(torch.LongTensor(test_sen).permute(1, 0).to(self.device))
        _, preds = torch.max(output, 1)
        return 'positive' if preds.cpu().numpy()[0] == 1 else 'negative'


if __name__ == '__main__':
    am = AttnModelManager()
    print(am.predict('I hate that'))