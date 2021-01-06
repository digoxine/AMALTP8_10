import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import datamaestro
from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text(encoding="UTF-8") if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text(encoding="UTF-8")), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")
    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)

def padding_tensor(sequences):
    """
    :param sequences: list of tensors
    :return:
    """
    ls_array_X = []
    ls_array_Y = []
    for i in range(len(sequences)):
        ls_array_X.append(torch.tensor(sequences[i][0]))
        ls_array_Y.append(torch.tensor(sequences[i][1]))

    print("ls_array_x : {} ".format(len(ls_array_X)))
    sequences = ls_array_X
    labels = torch.tensor(ls_array_Y)
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, labels

def custom_coll(sequences):
    ls_array_X = []
    ls_array_Y = []
    for i in range(len(sequences)):
        ls_array_X.append(torch.tensor(sequences[i][0]))
        ls_array_Y.append(torch.tensor(sequences[i][1]))
    myPadX = torch.nn.utils.rnn.pad_sequence(sequences=ls_array_X, batch_first=True,padding_value=-1)
    labels = torch.tensor(ls_array_Y, device=device)
    #print(labels)
    return myPadX,labels

print(datamaestro.__version__)
# 50 100 200 300
EMBEDDING_SIZE = 50
OUTPUT_SIZE = 2
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 10
word2id, embeddings, text_train, text_test = get_imdb_data(EMBEDDING_SIZE)
embeddings = torch.tensor(embeddings).to(device)
#print(type(embeddings))
train_loader = DataLoader(text_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_coll, drop_last=True)
#for x,y in train_loader:
#    print(len(x))
class BaseNetEmbedding(torch.nn.Module):
    def __init__(self):
        super(BaseNetEmbedding, self).__init__()
        self.linear = torch.nn.Linear(EMBEDDING_SIZE, OUTPUT_SIZE).to(device)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.relu   = torch.nn.ReLU().to(device)

    def forward(self, t):

        #print(t[0].shape)
        #print("len t : {}".format(len(t)))
        tmp = torch.zeros(size=(BATCH_SIZE,EMBEDDING_SIZE)).to(device)
        ls_res = []

        for batch in range(t.shape[0]):
            for wi in range(t.shape[1]):
                #print("embedding[wi] size {} tmp[batch] {}".format(embeddings[wi].shape, tmp[batch].shape))
                if (wi==-1):
                    continue
                tmp[batch] = tmp[batch] + embeddings[wi]
            ls_res.append(tmp[batch] / len(t))


        #print("ls_res {}".format(ls_res))
        res = torch.stack(ls_res).to(device)
        #print("res BaseNetEmbedding forward : {}".format(res.shape))
        res = self.linear(res.float())
        #print("res shape: {}".format(res.shape))
        return self.relu(res)

class AttentionSimple(torch.nn.Module):
    def __init__(self):
        super(AttentionSimple, self).__init__()
        self.linear = torch.nn.Linear(EMBEDDING_SIZE, OUTPUT_SIZE)


    def forward(self, q, k):
        tmp = torch.zeros(size=(BATCH_SIZE, EMBEDDING_SIZE)).to(device)
        ls_res = []
        pa_sum = []
        for batch in range(k.shape[0]):
            pa_i_t = []
            for wi in range(k.shape[1]):
                if(wi==-1):
                    continue

                res = q * embeddings[wi]
                print("res shape {}".format(res.shape))

        #return torch.nn.functional.softmax(self.linear(t))



def train():
    myNet = BaseNetEmbedding().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=myNet.parameters() ,lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss=0
        for x,y in train_loader:
            #print(y)
            #print(x.shape)
            x = x.to(device)
            y = y.to(device)
            y_pred = myNet(x)
            #print(y_pred.shape)
            #print(y.shape)
            loss = criterion(y_pred.view(-1, y_pred.shape[1]), y.view(-1)).to(device)
            optimizer.zero_grad()
            train_loss += loss.data.to(device).item()
            loss.backward()
            optimizer.step()
        print("train_loss mean {}".format(train_loss/(text_train.__len__()/BATCH_SIZE)))

def train2():
    myNet = AttentionSimple().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=myNet.parameters() ,lr=LEARNING_RATE)
    embeddingsQ = torch.nn.Embedding(OUTPUT_SIZE, EMBEDDING_SIZE).to(device)

    for epoch in range(EPOCHS):
        train_loss=0
        for x,y in train_loader:
            #print(y)
            #print(x.shape)
            x = x.to(device)
            y = y.to(device)
            q = embeddingsQ(y)
            print("Q shape {}".format(q.shape))
            print("q shape {}".format(q.shape))
            y_pred = myNet(q, x)
        print("train_loss mean {}".format(train_loss/(text_train.__len__()/BATCH_SIZE)))


train2()
#  TODO: 
