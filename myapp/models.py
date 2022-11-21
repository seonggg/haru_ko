import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kobertmodel.settings")

import django
django.setup()
from django.db import models

# Create your models here.
import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np

# kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


# KoBERT에 입력될 데이터셋 정리
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


# 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=4,  ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# BERT 모델, Vocabulary 불러오기 필수
bertmodel, vocab = get_pytorch_kobert_model()
bertmodel.return_dict=True

# Setting parameters
max_len = 64
batch_size = 128
warmup_ratio = 0.1
num_epochs = 4
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5


## 학습 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cuda:0")
PATH = 'C:/Users/rselo/Desktop/프종/saved_model/'
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
#model = torch.load(PATH + 'KoBERT_담화_4.pt', map_location='cpu')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + 'model_state_dict_4.pt', map_location=device))  # state_dict를 불러 온 후, 모델에 저장

# 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# 예측 모델 설정
def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        #label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                #result = "분노"
                result = "angry"
            elif np.argmax(logits) == 1:
                #result = "슬픔"
                result = "sad"
            elif np.argmax(logits) == 2:
                #result = "불안"
                result = "anxiety"
            elif np.argmax(logits) == 3:
                #esult = "기쁨"
                result = "happy"

        return result

from koalanlp.Util import initialize, finalize
from koalanlp.proc import SentenceSplitter
from koalanlp import API

def print_list(sentence):
    if sentence != "0":
        initialize(OKT='LATEST')
        splitter = SentenceSplitter(API.OKT)
        paragraph = splitter(sentence)
        emotion = []
        result=[]

        for num in paragraph:
            emotion.append(predict(num))

        emo_set=set(emotion)
        emotion=list(emo_set)
        result.append(emotion[0])
        result.append(emotion[1])
        result.append(emotion[2])

        finalize()
        return result

class Text(models.Model):
    text=models.TextField()

    def __str__(self):
        return self.text
