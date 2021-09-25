from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset , random_split , DataLoader , RandomSampler ,SequentialSampler
import pandas as pd
import numpy as np
import torch
from model import Bert_model
from train import train,test

batch_size = 32
def training_model():
    # 加载
    train_data = pd.read_csv('data/train_dataset_v2.tsv', sep='\t')
    print(f'Number of training tweets: {train_data.shape[0]}\n')
    train_data.dropna(inplace=True)
    # 数据提取
    labels = train_data['emotions'].values
    text = train_data['content'].values

    # 获得token
    input_ids = []
    max_len = 0
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for x in text:
        temp_ids = tokenizer.encode(x, add_special_tokens=True)
        max_len = max(max_len, len(temp_ids))
        input_ids.append(temp_ids)
    input_ids = np.array([i + [0]*(max_len-len(i)) for i in input_ids])
    attention_masks = np.where(input_ids != 0, 1, 0)

    # 获得labels
    y = []
    for label in labels:
        if pd.isna(label):
            y.append(np.array([0,0,0,0,0,0]))
        else:
            temp = np.array(list(map(int, label.split(','))))
            y.append(temp)

    # 截取所需
    input_ids = input_ids[:5000,:128]
    attention_masks = attention_masks[:5000,:128]
    labels = np.array(y)[:5000]

    # 数据集划分 - dataloader
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    print('{:>5,} test samples'.format(test_size))

    train_loader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )
    val_loader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    test_loader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    # model
    model =  Bert_model()

    # 训练

    train(model,train_loader,val_loader,test_loader)

def get_answer():
    # 测试
    test_data = pd.read_csv('./data/test_dataset.tsv', sep='\t')
    text = test_data['content'].values
    print(len(text))

    # 获得token
    input_ids = []
    max_len = 0
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for x in text:
        temp_ids = tokenizer.encode(x, add_special_tokens=True)
        max_len = max(max_len, len(temp_ids))
        input_ids.append(temp_ids)
    input_ids = np.array([i + [0]*(max_len-len(i)) for i in input_ids])
    attention_masks = np.where(input_ids != 0, 1, 0)

    # 截取所需
    input_ids = input_ids[:,:128]
    attention_masks = attention_masks[:,:128]

    #dataloader
    dataset = TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks)) #注意：test无lables 无法求损失 直接得到结果
    data_loader = DataLoader(
                dataset, # The validation samples.
                sampler = SequentialSampler(dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    #送人model
    model =  Bert_model()
    flat_predictions = test(model,data_loader)
    flat_predictions = [str(emotion.cpu().numpy())[1:-1].replace(' ', ',') for emotion in flat_predictions]
    # print(len(flat_predictions))
    # print(flat_predictions[0])

    #写入提交examble
    submit = pd.read_csv('./data/submit_example.tsv', sep='\t')
    submit ['emotion'] = flat_predictions
    submit.to_csv("./data/submission.tsv", index=False, sep='\t', encoding='utf-8')

def main():
    training_model()
    get_answer()

if __name__ == '__main__':
    main()