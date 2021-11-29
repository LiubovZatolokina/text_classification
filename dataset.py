import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.legacy.data import BucketIterator

BATCH_SIZE = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset():

    TEXT = data.Field(tokenize=data.get_tokenizer('basic_english'),
                      init_token='<SOS>', eos_token='<EOS>', lower=True)
    LABEL = data.LabelField(dtype=torch.long)
    legacy_train, legacy_test = datasets.IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(legacy_train)
    LABEL.build_vocab(legacy_train)
    train_loader, test_loader = BucketIterator.splits((legacy_train, legacy_test), batch_size=BATCH_SIZE,
                                                      device=device)
    return TEXT, LABEL, train_loader, test_loader

