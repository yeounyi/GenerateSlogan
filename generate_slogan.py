import pandas as pd
import argparse
import random
import torch.nn as nn
import numpy as np
import torch
from transformers import RobertaModel, RobertaConfig, logging
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaLMHead
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM
from transformers import TrainingArguments, Trainer
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
import ast
from datasets import load_metric
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict

pd.set_option('display.max_colwidth', -1)

logging.set_verbosity_info()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_slogan(input_keyword):

    # https://github.com/godatadriven/rhyme-with-ai/blob/master/src/rhyme_with_ai/rhyme.py
    def query_datamuse_api(word: str, n_rhymes) -> list:
        """Query the DataMuse API.
        Parameters
        ----------
        word : Word to rhyme with
        n_rhymes : Max rhymes to return
        Returns
        -------
        ml: meaning like & sl: sound like
        """

        out = requests.get(
            "https://api.datamuse.com/words", params={"ml": word, "sl":word}
        ).json()

        #out = requests.get(
        #    "https://api.datamuse.com/words", params={"ml": word, "rel_cns":word}
        #).json()

        words = [_["word"] for _ in out]
        if n_rhymes is None:
            return words
        return words[:n_rhymes]

    candidates = query_datamuse_api(input_keyword, 5)
    keywords = []
    for c in candidates:
        keywords.append([input_keyword, c])
        keywords.append([c, input_keyword])

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, input_words, tokenizer, labels=None):
            self.labels = labels

            inputs = tokenizer(input_words, return_tensors="pt", padding=True, truncation=True)
            self.input_ids = [torch.tensor(i).long() for i in inputs.input_ids]

            token_type_ids = []
            padded_length = len(self.input_ids[0])
            # batch_size?????? ??????
            for i in range(len(self.input_ids)):
                key1, key2 = input_words[i].split()
                len_key1 = len(tokenizer(' ' + key1).input_ids[1:-1])  # [CLS], [SEP] ??????
                len_key2 = len(tokenizer(' ' + key2).input_ids[1:-1])
                # len_key1 +1 : [CLS] ????????? +1
                token_type_id = torch.cat((torch.zeros(len_key1 + 1), torch.ones(len_key2)))
                padded_token_type_id = torch.cat((token_type_id, torch.zeros(padded_length - len(token_type_id))))
                token_type_ids.append(padded_token_type_id)

            # ????????? ????????? ????????? ??????????????? id
            self.token_type_ids = [torch.tensor(i).long() for i in token_type_ids]

            self.attention_mask = [torch.tensor(i) for i in inputs.attention_mask]

            if labels:
                self.labels = [torch.tensor(label).long() for label in labels]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            if self.labels:
                return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], \
                        'token_type_ids': self.token_type_ids[idx], 'labels': self.labels[idx]}
            else:
                return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], \
                        'token_type_ids': self.token_type_ids[idx]}


    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.add_tokens('<name>')
    tokenizer.add_tokens('<loc>')
    tokenizer.add_tokens('<year>')

    input_words = [' ' + ' '.join(k) for k in keywords]

    data = CustomDataset(input_words, tokenizer)

    class MaskClassifier(RobertaPreTrainedModel):
        def __init__(self, config):
            super().__init__(config=config)
            self.roberta = RobertaModel(config)
            self.max_mask = 5
            self.hidden_size = config.hidden_size
            self.linear1 = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.batchnorm = torch.nn.BatchNorm1d(num_features=self.hidden_size)
            self.linear2 = torch.nn.Linear(self.hidden_size, self.max_mask + 1)
            self.softmax = torch.nn.Softmax(dim=1)
            self.dropout = torch.nn.Dropout(0.5)
            self.relu = torch.nn.ReLU()
            self.init_weights()

        def forward(self, input_ids, attention_mask, token_type_ids, labels=None):

            # Feed input to RoBERTa
            # roberta??? token_type_ids ????????? ????????? (?????????)
            # outputs = self.roberta(input_ids, attention_mask, token_type_ids)
            outputs = self.roberta(input_ids, attention_mask)

            # last hidden state: (batch_size, sequence_length, hidden_size)
            org_hidden = outputs[0]
            batch_size = org_hidden.shape[0]

            # ????????? ???????????? tokenizer?????? ???????????? ?????? ??????, ?????? hidden size ????????? ?????? (mean)
            # ?????? ?????? tensor??? gpu??? ??? ???????????? ?????? ??? ???
            hidden = torch.zeros((batch_size, 4, self.hidden_size)).to(device)
            hidden[:, 0, :] = org_hidden[:, 0, :]  # [CLS] hidden

            # ???????????? 1??? ????????? index
            first_1_idx = [list(t).index(1) for t in token_type_ids]
            # 2?????? 0, ??? [PAD] ?????? [SEP]??? ???????????? index
            sec_0_idx = [len(t) - 1 - list(t)[::-1].index(1) + 1 for t in token_type_ids]

            for i in range(batch_size):
                hidden[i, 1, :] = torch.mean(org_hidden[i, 1:first_1_idx[i], :], dim=0)
                hidden[i, 2, :] = torch.mean(org_hidden[i, first_1_idx[i]:sec_0_idx[i], :], dim=0)
                hidden[i, 3, :] = torch.mean(org_hidden[i, sec_0_idx[i]:, :], dim=0)

            # logits: (batch_size, 3, 11)
            # 3??? mask_num??? ?????? 3??? ????????????
            # 11??? 0-10?????? ??????
            logits = torch.zeros((batch_size, 3, self.max_mask + 1)).to(device)

            for i in range(3):
                concated_h = torch.cat((hidden[:, i, :], hidden[:, i + 1, :]), dim=1)  # batch_size, 2*hidden_size
                # y = self.dropout(self.relu(self.batchnorm(self.linear1(concated_h))))
                # y = self.dropout(self.relu(self.linear1(concated_h)))
                # y = self.softmax(self.linear2(y))
                # logits[:, i] = y
                logits[:, i] = self.softmax(self.linear2(self.linear1(concated_h)))

            if labels is not None:
                labels = labels.to(device)

                # argmax??? ?????? ????????? -> loss??? ?????? ?????????
                criterion = CrossEntropyLoss()
                # logits.permute(0,2,1): (batch_size, max_mask+1, 3)
                # labels: (batch_size, 3)
                loss = criterion(logits.permute(0, 2, 1), labels)

                # tuple of (loss, prediction)
                # loss, if labels provided
                return (loss, logits)

            else:
                return logits

    model = MaskClassifier.from_pretrained('pretrained_mask_ins0209')
    model = model.to(device)

    def compute_metrics(preds):
        labels = torch.tensor(preds.label_ids)
        predictions = torch.argmax(torch.tensor(preds.predictions), dim=2)
        pdist = torch.nn.PairwiseDistance(p=2)

        N = len(labels)
        # averaged distance between labels and predictions
        distance = sum(pdist(labels, predictions)) / N

        exact = 0
        for i in range(N):
            # 3?????? ?????? ?????? ????????? True 3??? -> ?????? 3
            if sum(labels[i] == predictions[i]) == 3:
                exact += 1
        exact_match = exact / N

        df = pd.DataFrame(columns=['label', 'pred'])
        df['label'] = list(labels)
        df['pred'] = list(predictions)
        df.to_csv('mask_ins_eval.csv', index=False, encoding='utf-8')

        return {
            'distance': distance,
            'exact_match': exact_match,
        }

    training_args = TrainingArguments(
        num_train_epochs=100,  # total # of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        output_dir='./MASKckpt'
    )

    trainer = Trainer(
        model=model,  # the instantiated ???? Transformers model to be trained
        args=training_args,  # training arguments, defined above
        #train_dataset=train_data,  # training dataset
        #eval_dataset=eval_data,
        compute_metrics=compute_metrics
    )

    mask_ins_result = torch.tensor(trainer.predict(data).predictions).argmax(dim=2) # (batch size, 3)


    def mask_num2token(keywords, mask_num):
        key1, key2 = keywords
        mask_seq = '<mask> ' * mask_num[0] + key1 + ' ' + '<mask> ' * mask_num[1] + key2 + ' ' + '<mask> ' * mask_num[2]
        if mask_seq.endswith(' '):
            mask_seq = mask_seq[:-1] # ??? ??? ?????? ?????????
        return mask_seq

    mask_seq = []
    for k, m in zip(keywords, mask_ins_result):
        mask_seq.append(mask_num2token(k, m))


    class CustomDataset2(torch.utils.data.Dataset):
        def __init__(self, inputs, tokenizer, labels=None):
            inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
            self.input_ids = [torch.tensor(i) for i in inputs.input_ids]
            self.attention_mask = [torch.tensor(i) for i in inputs.attention_mask]
            if labels:
                labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).input_ids
                self.labels = [torch.tensor(i) for i in labels]

        def __len__(self):
            assert len(self.input_ids) == len(self.attention_mask) == len(self.labels)

            return len(self.input_ids)

        def __getitem__(self, idx):
            return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'labels': self.labels[idx]}

    data = CustomDataset2(mask_seq, tokenizer)

    class MaskedLM(RobertaPreTrainedModel):
        def __init__(self, config):
            super().__init__(config=config)
            self.roberta = RobertaModel(config)
            self.lm_head = RobertaLMHead(config)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.tokenizer.add_tokens('<name>')
            self.tokenizer.add_tokens('<loc>')
            self.tokenizer.add_tokens('<year>')
            self.refinement_num = 10
            self.mask_id = 50264
            # self.mask_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token])[0] # 50264
            self.init_weights()

        def forward(self, input_ids, attention_mask, labels=None):

            batch_size = input_ids.size(0)

            if labels is not None:
                labels = labels.to(device)

            # labels ????????? argument??? ???????????? ???
            outputs = self.roberta(input_ids, attention_mask)
            prediction_scores = self.lm_head(outputs[0])

            # logits, hidden_states, attentions
            outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.roberta.config.vocab_size), labels.view(-1))
                outputs = (masked_lm_loss,) + outputs  # loss, logits, hidden_states, attentions

            # keyword_idxes: (batch_size, 2)
            # keyword_idx = self.check_keyword(input_ids)

            # predictions: (batch_size, sequence_length)
            predictions = prediction_scores.argmax(dim=2)

            # print(self.tokenizer.decode(input_ids[0]))
            # print(self.tokenizer.decode(predictions[0]))

            # REFINEMENT
            for n in range(self.refinement_num):
                for b in range(batch_size):
                    # print(self.tokenizer.decode(input_ids[b]))
                    # print(self.tokenizer.decode(predictions[b]))
                    keyword_dict = {idx: iid for idx, iid in enumerate(input_ids[b]) if iid not in [0, 1, 2, 50264]}
                    for key in keyword_dict:
                        predictions[b][key] = keyword_dict[key]  # keyword ??????
                    remask_idx = self.find_refinable(predictions[b], list(keyword_dict.keys()), attention_mask[b])
                    predictions[b][remask_idx] = self.mask_id
                    # print(self.tokenizer.decode(predictions[b]))

                outputs = self.roberta(predictions, attention_mask)
                prediction_scores = self.lm_head(outputs[0])
                predictions = prediction_scores.argmax(dim=2)

                # print(self.tokenizer.decode(predictions[0]))

                outputs = (prediction_scores,) + outputs[2:]
                if labels is not None:
                    masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                    outputs = (masked_lm_loss,) + outputs  # loss, logits, hidden_states, attentions

            return outputs

        def find_refinable(self, prediction, keyword_idx, attention_mask):
            # keyword??? ?????????, <sos>, <pad>??? ????????? attention_mask??? 0??? ????????? ?????? ????????? random?????? ?????? ?????????
            # ????????? <sos>??? ???????????? ????????? ????????? ??? ??? ?????? idx??? ????????? ??? ????????? ?????? ??????
            # token != 1 ?????? <pad> ????????? ????????? ????????? attention_mask??? ?????????
            # <mask>??? <eos>??? ???????????? ????????? ????????? token??? 2??? ????????? refine ??????????????? ???
            # <pad> ?????? ????????? ????????? attention_mask??? ?????? 0 ?????? ?????? ??????
            # list(attention_mask).index(0)-1: </s>??? ?????? ????????? ????????? -1
            refinable_idx = [idx for idx, token in enumerate(prediction) if idx != 0 \
                             and idx not in keyword_idx and \
                             (0 not in list(attention_mask) or idx < list(attention_mask).index(0) - 1)]

            return random.choice(refinable_idx)

        def get_output_embeddings(self):
            return self.lm_head.decoder

        def set_output_embeddings(self, new_embeddings):
            self.lm_head.decoder = new_embeddings

    model = MaskedLM.from_pretrained('pretrained_mask_lm_refined0223')
    model = model.to(device)


    input_ids = torch.stack(data.input_ids)
    attention_mask = torch.stack(data.attention_mask)

    preds = model(input_ids.to(device), attention_mask.to(device))


    pred = []
    for i,logit in enumerate(preds[0]):
        keyword_dict = {idx:iid for idx, iid in enumerate(input_ids[i]) if iid not in [0, 1, 2, 50264]}
        raw_pred = logit.argmax(dim=1)
        for k in keyword_dict:
            raw_pred[k] = keyword_dict[k]
        pred.append((tokenizer.decode(raw_pred)).replace('<s>', '').replace('</s>', '').replace('<pad>', ''))

    return pred
    #print(pred)


def score_slogan(pred):
    tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    tokenizer.add_tokens('<name>')
    tokenizer.add_tokens('<loc>')
    tokenizer.add_tokens('<year>')
    model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")
    model.resize_token_embeddings(len(tokenizer))

    inputs = tokenizer(pred, return_tensors="pt", padding=True, truncation=True)
    outputs = model(inputs.input_ids, inputs.attention_mask)

    grammar_score = outputs.logits
    grammar_total_score = torch.matmul(torch.tensor([-1,1], dtype=torch.long), torch.transpose(grammar_score.long(),0,1)) # (1,10)
    # print(grammar_total_score) # (1,10)
    # torch.argsort(grammar_total_score) # ??? ?????? ???????????? index tensor ??????

    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    tokenizer.add_tokens('<name>')
    tokenizer.add_tokens('<loc>')
    tokenizer.add_tokens('<year>')
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model.resize_token_embeddings(len(tokenizer))

    inputs = tokenizer(pred, return_tensors="pt", padding=True, truncation=True)
    outputs = model(inputs.input_ids, inputs.attention_mask)

    senti_score = outputs.logits
    senti_total_score = torch.matmul(torch.tensor([-2,-1,0,1,2], dtype=torch.long), torch.transpose(senti_score.long(),0,1)) # (1,10)
    # print(senti_total_score) # (1,10)
    # torch.argsort(senti_total_score) # ??? ?????? ???????????? index tensor ??????


    final_dict = defaultdict(lambda:0)
    for i, key in enumerate(torch.argsort(grammar_total_score)):
        final_dict[int(key)] += i

    for i, key in enumerate(torch.argsort(senti_total_score)):
        final_dict[int(key)] += i

    # sort dict by highest value
    pred_score = sorted(final_dict, key=final_dict.get, reverse=True)

    sorted_slogans = []
    for i, score in enumerate(pred_score):
        sorted_slogans.append(pred[score])

    # showing only top 5
    return sorted_slogans[:5]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-keyword', dest='keyword', help='input keyword to generate slogan with')
    args = parser.parse_args()

    if args.keyword:
        pred = generate_slogan(str(args.keyword))
        scored_pred = score_slogan(pred)
        print()
        print('Ta-Da ????????')
        for i in range(len(scored_pred)):
            print(i+1, ' ' + scored_pred[i])

# CUDA_VISIBLE_DEVICES=0 python3 generate_slogan.py -keyword cake
