# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 


# [Processing the data]
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# 
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
optimizer.zero_grad()
loss = model(**batch).loss
loss.backward()
optimizer.step()

# [loading a dataset from the Hub]
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
# 
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
# 特别注意label
print(raw_train_dataset.features)  

# [Preprocessing a dataset]
from transformers import AutoTokenizer
print(len(raw_train_dataset["sentence1"]))

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_train_dataset["sentence1"])
tokenized_sentences_2 = tokenizer(raw_train_dataset["sentence2"])

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)
"""
{'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
"""

outputs = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
print(outputs)

outputs = tokenizer.decode(inputs["input_ids"])
print(outputs)
"""
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 
'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
"""
# 
tokenized_dataset = tokenizer(
    raw_train_dataset["sentence1"],
    raw_train_dataset["sentence2"],
    padding=True,
    truncation=True,
)
tokenized_dataset.keys()
"""
dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
"""


# 只需要填充批处理的最长长度.
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

# [Dynamic padding]
from transformers import DataCollatorWithPadding
# padding token
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])
batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})

# [Fine-tuning a model with the Trainer API]

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# [Training]
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
from transformers import AutoModelForSequenceClassification

# 去旧头, 加新头
# 警告: 一些权重未使用, 一些权重随机初始化
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    # data_collator=data_collator,  # 默认
    tokenizer=tokenizer,
)

trainer.train()
# 
predictions = trainer.predict(tokenized_datasets["validation"])  # namedtuple
print(predictions.predictions.shape, predictions.label_ids.shape)
print(predictions._fields)
"""
(408, 2) (408,)
('predictions', 'label_ids', 'metrics')
"""

import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)
from datasets import load_metric

metric = load_metric("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
"""
{'accuracy': 0.8357843137254902, 'f1': 0.8838821490467937}
"""
# [Evaluation]

def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
"""
TrainOutput(global_step=1377, 
training_loss=0.339369716381109, 
metrics={'train_runtime': 95.8143, 'train_samples_per_second': 114.847, 'train_steps_per_second': 14.372, 'train_loss': 0.339369716381109, 'epoch': 3.0})
"""

# [A full training]

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# [Prepare for training]
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")  # 默认List
tokenized_datasets["train"].column_names

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})
"""
{'labels': torch.Size([8]),
 'input_ids': torch.Size([8, 68]),
 'token_type_ids': torch.Size([8, 68]),
 'attention_mask': torch.Size([8, 68])}
"""


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
"""
tensor(0.6796, grad_fn=<NllLossBackward0>) torch.Size([8, 2])
"""
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)




# [The training loop]
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

from tqdm import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        # https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/01-introduction-to-pytorch.html
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)



# [The evaluation loop]
from datasets import load_metric

metric = load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
"""
{'accuracy': 0.8602941176470589, 'f1': 0.9015544041450777}
"""



# [Supercharge your training loop with Accelerate]

# from accelerate import Accelerator
# from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
# accelerator = Accelerator()
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# optimizer = AdamW(model.parameters(), lr=3e-5)

# print(accelerator.device)
# train_dl, eval_dl, model, optimizer = accelerator.prepare(
#     train_dataloader, eval_dataloader, model, optimizer
# )
# num_epochs = 3
# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,
# )

# progress_bar = tqdm(range(num_training_steps))

# model.train()
# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         outputs = model(**batch)
#         loss = outputs.loss
#         accelerator.backward()

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)
