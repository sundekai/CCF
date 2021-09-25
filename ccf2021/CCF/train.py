from torch.utils.data import TensorDataset , DataLoader  ,SequentialSampler
from transformers import AdamW
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm

model_weights_path = './models'

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def RMSELoss(predict, label):
    return torch.sqrt(torch.mean((predict-label)**2))

def run_epoch_train(model, iterator, class_optimizer, bert_optimizer,criterion):
    model.train()
    epoch_loss = 0
    for batch in tqdm(iterator):
        labels = batch[-1].to(get_device())
        class_optimizer.zero_grad()  # 把梯度重置为零
        bert_optimizer.zero_grad()
        predictions = model(batch)

        loss = 0
        for k in range(6):
            loss += criterion(predictions[:, k, :], labels[:, k].long())
        loss /= 6

        loss.backward()

        class_optimizer.step()  # 更新模型
        bert_optimizer.step()
        epoch_loss += loss

    average_loss = epoch_loss / len(iterator)
    return average_loss


def run_epoch_eval(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(iterator):
            labels = batch[-1].to(get_device())
            predictions = model(batch)
            loss = 0
            for k in range(6):
                loss += criterion(predictions[:, k, :], labels[:, k].long())
            loss /= 6
            epoch_loss += loss

    average_loss = epoch_loss / len(iterator)
    return average_loss

def train(model,train_loader,val_loader,test_loader):

    #定义优化器&损失函数
    calss_optimizer = torch.optim.Adam(model.calssification.parameters(), lr=0.001)
    bert_optimizer = AdamW(model.bert.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()

    train_loss = 999
    best_val_loss = 999
    n_epochs = 5
    patience = 10
    epochs_without_improvement = 0

    model = model.to(get_device())
    for epoch in range(n_epochs):

        if epochs_without_improvement == patience:
            break

        val_loss = run_epoch_eval(model, val_loader, criterion)
        if val_loss < best_val_loss:
            model.save(model_weights_path)
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print(
                "!Epoch: {} | Val loss improved to {:.4f} | train loss: {:.4f} | saved model to {}.".format(
                    epoch, best_val_loss,  train_loss, model_weights_path
                ))

        train_loss  = run_epoch_train(model, train_loader, calss_optimizer,bert_optimizer,criterion)


        epochs_without_improvement += 1

        print('Epoch: {} | Val Loss: {:.3f} | Train Loss: {:.4f}. '.format(epoch + 1, val_loss, train_loss))

    model.load(model_weights_path)
    model.to(get_device())
    test_loss = run_epoch_eval(model, test_loader, criterion)

    result = f'| Epoch: {epoch + 1} | Test Loss: {test_loss:.3f} '
    print(result)


# 真正的测试，获得提交答案，还没写好
def test(model,  iterator):
    #加载最优模型
    model.load(model_weights_path)
    model.to(get_device())
    #获取prediction
    model.eval()
    all_predict = []
    with torch.no_grad():
        for batch in tqdm(iterator):
            predictions = model(batch)
            predict = None
            for k in range(6):
                temp = torch.argmax(predictions[:, k, :],1)
                temp=temp.view(temp.shape[0],1)
                if k==0:
                    predict = temp
                else:
                    predict = torch.cat((predict,temp),1)
            all_predict.append(predict)
    flat_predictions = [ emotion for predict in all_predict for emotion in predict ]
    return  flat_predictions


