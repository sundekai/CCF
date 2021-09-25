from transformers import BertModel,WEIGHTS_NAME, CONFIG_NAME
import torch
import os
from train import get_device

class FCModel(torch.nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.name = "classification"
        self.fc1 = torch.nn.Linear(in_features=768, out_features=4)
        self.fc2 = torch.nn.Linear(in_features=768, out_features=4)
        self.fc3 = torch.nn.Linear(in_features=768, out_features=4)
        self.fc4 = torch.nn.Linear(in_features=768, out_features=4)
        self.fc5 = torch.nn.Linear(in_features=768, out_features=4)
        self.fc6 = torch.nn.Linear(in_features=768, out_features=4)

    def forward(self, input):
        result1 = self.fc1(input)
        result2 = self.fc2(input)
        result3 = self.fc3(input)
        result4 = self.fc4(input)
        result5 = self.fc5(input)
        result6 = self.fc6(input)
        b = result1.shape[0]
        result = torch.cat((result1.view(b,1,4), result2.view(b,1,4), result3.view(b,1,4), result4.view(b,1,4),
                            result5.view(b,1,4), result6.view(b,1,4)),1)
        return  result

    def save(self, SAVE_PATH):
        torch.save(self.state_dict(), os.path.join(SAVE_PATH, '{}.pt'.format(self.name)))
    def load(self, LOAD_PATH):
        self.load_state_dict(torch.load(os.path.join(LOAD_PATH, '{}.pt'.format(self.name))))

class Bert_model(torch.nn.Module):
    def __init__(self,):
        super(Bert_model, self).__init__()
        self.name = "BERT"
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.calssification = FCModel()

    def forward(self, batch):
        input_ids = batch[0]
        attention_masks = batch[1]
        input_ids = torch.tensor(input_ids).to(get_device())
        attention_masks = torch.tensor(attention_masks).to(get_device())

        bert_output = self.bert(input_ids, attention_mask=attention_masks)
        pooler_output = bert_output[0][:,0]  # <==>  bert_output.hidden_states[0][:,0,:]
        predict = self.calssification(pooler_output).squeeze()
        return predict

    def load(self,output_dir):
        BERT_model_path = os.path.join(output_dir, 'BERT')
        CLASS_model_path = os.path.join(output_dir, 'CLASS')

        self.bert = BertModel.from_pretrained(BERT_model_path)
        self.calssification.load(CLASS_model_path)

    def save(self,output_dir):

        BERT_model_path = os.path.join(output_dir, 'BERT')
        if not os.path.exists(BERT_model_path ):
            os.makedirs(BERT_model_path )

        model_to_save = self.bert.module if hasattr(self.bert, 'module') else self.bert
        torch.save(model_to_save.state_dict(), os.path.join(BERT_model_path,WEIGHTS_NAME))
        model_to_save.config.to_json_file(os.path.join(BERT_model_path,CONFIG_NAME))

        CLASS_model_path = os.path.join(output_dir, 'CLASS')
        if not os.path.exists(CLASS_model_path):
            os.makedirs(CLASS_model_path)

        self.calssification.save(CLASS_model_path)


