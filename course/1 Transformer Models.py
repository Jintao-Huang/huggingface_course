# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

# [Transformers, what can they do?]


# [Transformers are everywhere!]
# [Working with pipelines]

from transformers import pipeline

# distilbert-base-uncased-finetuned-sst-2-english
classifier = pipeline("sentiment-analysis", device=0)
result = classifier(
    "I've been waiting for a HuggingFace course my whole life.")
print(result)


results = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"])
print(results)
"""
[{'label': 'POSITIVE', 'score': 0.9598415493965149}]
[{'label': 'POSITIVE', 'score': 0.9598415493965149}, {'label': 'NEGATIVE', 'score': 0.9994557499885559}]
"""


# [Zero-shot classification]

# bart-large-mnli
classifier = pipeline("zero-shot-classification", device=0)
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print(result)
"""
{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8450096249580383, 0.11171336472034454, 0.04327699542045593]}
"""

# [Text generation]
# gpt2
generator = pipeline("text-generation", device=0)
result = generator("In this course, we will teach you how to")
print(result)  # 每次打印的结果都不同

#
result = generator("In this course, we will teach you how to",
                   num_return_sequences=2, max_length=15)
print(result)  # 含提示词+标点=15

# [Using any model from the Hub in a pipeline]
# "distilgpt2"
generator = pipeline("text-generation", model="distilgpt2", device=0)
result = generator(
    "In this course, we will teach you how to",
    num_return_sequences=2,
    max_length=30,
)
print(result)

# [The Inference API]

# [Mask filling]
# distilroberta-base
unmasker = pipeline("fill-mask", device=0)
result = unmasker(
    "This course will teach you all about <mask> models.", top_k=2)

print(result)
"""
[{'score': 0.19633805751800537, 'token': 30412, 'token_str': ' mathematical', 
'sequence': 'This course will teach you all about mathematical models.'}, 
{'score': 0.04051966220140457, 'token': 38163, 'token_str': ' computational', 
'sequence': 'This course will teach you all about computational models.'}]
"""

# [Named entity recognition]
# bert-large-cased-finetuned-conll03-english
ner = pipeline("ner", device=0)
result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

print(result)
"""
[{'entity': 'I-PER', 'score': 0.999383, 'index': 4, 'word': 'S', 'start': 11, 'end': 12}, 
{'entity': 'I-PER', 'score': 0.99815637, 'index': 5, 'word': '##yl', 'start': 12, 'end': 14}, 
{'entity': 'I-PER', 'score': 0.99591184, 'index': 6, 'word': '##va', 'start': 14, 'end': 16}, 
{'entity': 'I-PER', 'score': 0.9992335, 'index': 7, 'word': '##in', 'start': 16, 'end': 18}, 
{'entity': 'I-ORG', 'score': 0.9738477, 'index': 12, 'word': 'Hu', 'start': 33, 'end': 35}, 
{'entity': 'I-ORG', 'score': 0.9761318, 'index': 13, 'word': '##gging', 'start': 35, 'end': 40}, 
{'entity': 'I-ORG', 'score': 0.98882234, 'index': 14, 'word': 'Face', 'start': 41, 'end': 45}, 
{'entity': 'I-LOC', 'score': 0.9932112, 'index': 16, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
"""

# [Question answering]
# distilbert-base-cased-distilled-squad
question_answerer = pipeline("question-answering", device=0)
result = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(result)
"""
{'score': 0.6949892640113831, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
"""


# [Summarization]
# distilbart-cnn-12-6 
summarizer = pipeline("summarization", device=0)
result = summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
, max_length=50)
print(result)
"""
[{'summary_text': ' America has changed dramatically during recent years . 
The number of engineering graduates in the U.S. has declined in traditional engineering disciplines 
such as mechanical, civil,    electrical, chemical, and aeronautical engineering . 
Rapidly developing economies such as China and India continue to encourage 
and advance the teaching of engineering .'}]
"""

# [Translation]
# distilbart-cnn-12-6

# sentencepiece
translator = pipeline(
    "translation", model="Helsinki-NLP/opus-mt-fr-en", device=0)
result = translator("Ce cours est produit par Hugging Face.")
print(result)
"""
[{'translation_text': 'This course is produced by Hugging Face.'}]
"""


# [Bias and limitations]
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])




