# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from transformers import pipeline

classifier = pipeline("sentiment-analysis", device=0)
result = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
"""
[{'label': 'POSITIVE', 'score': 0.9598415493965149}, 
{'label': 'NEGATIVE', 'score': 0.9994557499885559}]

"""

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
"""
{'input_ids': 
tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
             0,     0,     0,     0,     0,     0]]), 
'attention_mask': 
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
"""


from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.keys())
print(outputs.last_hidden_state.shape)
"""
odict_keys(['last_hidden_state'])
torch.Size([2, 16, 768])
"""

# [namedtuple; NameSpace]

from types import SimpleNamespace
from collections import namedtuple

sn = SimpleNamespace(x=1, y=2)
print(sn.x, sn.y)
NT = namedtuple("nt", ["x", "y"])
nt = NT(x=1, y=2)
print(nt.x, nt.y)
# 

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.keys())
print(outputs.logits.shape)
"""
odict_keys(['logits'])
torch.Size([2, 2])
"""

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

"""
tensor([[4.0195e-02, 9.5981e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward0>)
"""

print(model.config.id2label)
"""
{0: 'NEGATIVE', 1: 'POSITIVE'}
"""

# [Models]
# [Creating a Transformer]
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)
print(config)
# [Different loading methods]
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)  # 随机初始化

from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")
# .json .bin

# BertForSequenceClassification

# [saving methods]


model.save_pretrained("directory_on_my_computer")



# [Using a Transformer model for inference]
sequences = ["Hello!", "Cool.", "Nice!"]
# start token; end token
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
import torch

model_inputs = torch.tensor(encoded_sequences)

# [Using the tensors as inputs to the model]

output = model(model_inputs)
print(output.keys())
print(output.last_hidden_state)
print(output.pooler_output)
"""
torch.Size([3, 4, 768])
torch.Size([3, 768])
"""
# [Tokenizers]


# [Word-based]
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)

# [Character-based]


# [Subword tokenization]


# [And more!]



# [Loading and saving]
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

outputs = tokenizer("Using a Transformer network is simple")
print(outputs)

tokenizer.save_pretrained("directory_on_my_computer")
"""
{'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

"""
# [Encoding]



# [Tokenization]
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
"""
['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
"""

# [From tokens to input IDs]
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
"""
[7993, 170, 13809, 23763, 2443, 1110, 3014]
"""
outputs = tokenizer(sequence)
print(outputs["input_ids"])

"""
{'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""
# [Decoding]

decoded_string = tokenizer.decode(outputs["input_ids"])
print(decoded_string)
"""
[CLS] Using a Transformer network is simple [SEP]
"""


# [Handling multiple sequences]

# [Models expect a batch of inputs]
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)
model(input_ids)
# 
input_ids = input_ids[None, :]
model(input_ids)
# 
tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
"""
[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102]])
"""
# 

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

# 
batched_ids = [ids, ids]
input_ids = torch.tensor(batched_ids)
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

# [Padding the inputs]
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
"""
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
tensor([[ 1.5694, -1.3895],
        [ 1.3374, -1.2163]], grad_fn=<AddmmBackward0>)
"""

# [Attention masks]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
"""
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
"""


# [Longer sequences]
max_sequence_length = 512
sequence = sequence[:max_sequence_length]


# [Putting it all together]
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 
sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
print(model_inputs)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs = tokenizer(sequences)
print(model_inputs)



# pad
model_inputs = tokenizer(sequences, padding="longest")
print(model_inputs)
model_inputs = tokenizer(sequences, padding="max_length")
print(model_inputs)
model_inputs = tokenizer(sequences, padding="max_length", max_length=8,)
print(model_inputs)
# truncate

model_inputs = tokenizer(sequences, truncation=True)
print(model_inputs)
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
print(model_inputs)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
print(model_inputs)
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")
print(model_inputs)
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
print(model_inputs)
"""
{'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102],
        [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0]]), 
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}

{'input_ids': <tf.Tensor: shape=(2, 16), dtype=int32, numpy=
array([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662,
        12172,  2607,  2026,  2878,  2166,  1012,   102],
       [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0]], dtype=int32)>, 
'attention_mask': <tf.Tensor: shape=(2, 16), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>}
{'input_ids': array([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662,
        12172,  2607,  2026,  2878,  2166,  1012,   102],
       [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0]]), 
'attention_mask': array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}

"""

# [Special tokens]
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))

# [Wrapping up: From tokenizer to model]
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
print(output)

"""
SequenceClassifierOutput(loss=None, logits=tensor([[-1.5607,  1.6123],
        [-3.6183,  3.9137]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
"""

# 

model_inputs = tokenizer(sequences, padding="max_length", truncation=True, max_length=8, return_tensors="pt")
print(model_inputs)
"""
{'input_ids': tensor([[ 101, 1045, 1005, 2310, 2042, 3403, 2005,  102],
        [ 101, 2061, 2031, 1045,  999,  102,    0,    0]]), 
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0]])}
"""