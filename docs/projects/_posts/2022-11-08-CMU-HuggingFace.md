---
layout: post
title:  "HuggingFace: What's it For?"
date:   2022-11-08 18:19:00 -0400
categories: CS, python, personal, ML, AI, HuggingFace
---

## Overview
[HuggingFace](https://huggingface.co/) is a rapidly developing tool with lots of hype and investment ([$2 billion in May 2022](https://techcrunch.com/2022/05/09/hugging-face-reaches-2-billion-valuation-to-build-the-github-of-machine-learning/)), so let's take some time to discuss:
1. How can HuggingFace help *me*?
2. How do I use it?
3. What are some strengths and limitations?


## How can HuggingFace help *me*?
First, I'll start by addressing the *me*. Generally these are Data Scientists and ML Engineers because,broadly speaking, HuggingFace provides "models, datasets, ML demos, and libraries"([https://huggingface.co/about](HuggingFace-About)); however, they even have [no code solutions](https://huggingface.co/autotrain) for the less programming savvy.

To me, a particularly powerful component of HuggingFace is the "one stop shop" feeling, which makes rapid ideation and iteration seamless. Furthermore, [HuggingFace Spaces](https://huggingface.co/spaces) provides a great way to demo models. The API and Spaces will be discussed more below, but keep in mind that HuggingFace is evolving and expanding very quickly.

## How do I use it?
To demonstrate the usefullness, I'll work on the example of clustering movie genres. To do this, I use Natural Language Processing (NLP) to embed the genre into a latent space and then simple k-means to cluster the movies. 

To use the NLP models on HuggingFace, I start by installing the [HuggingFace Transformers](https://huggingface.co/docs/transformers/main/en/index) package. Now, after a cursory search for some popular architectures, I decide to experiment with: ```distilbert-base-uncased, bert-base-uncased, bert-base-cased, roberta-base, xlm-roberta-base```
This seems like a hard task (each of these models is trained differently, written with different structures and frameworks, from completely different institutions, etc.), but the HuggingFace interface simplifies this down to the simple python snippet below:
```Python
from transformers import DistilBertTokenizer, DistilBertModel, \
                         BertTokenizer, BertModel, \
                         RobertaTokenizer, RobertaModel, \
                         AutoTokenizer, AutoModelForMaskedLM

def update_models(current_encoder: str) -> None:
    if current_encoder == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    elif current_encoder == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    elif current_encoder == 'bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = BertModel.from_pretrained('bert-base-cased')
    elif current_encoder == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
    elif current_encoder == 'xlm-roberta-base':
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
```
Now switching out the NLP models is completely painless. Furthermore, these all share common methods so embedding text is as simple as running:
```Python
encoded_input = tokenizer(text, return_tensors='pt')
new_output = model(**encoded_input)
```

This is awesome! But to really sell the project, I put together a quick demo using [gradio](gradio.app/docs/). Now I hosted this demo on my very own space [here!](https://huggingface.co/spaces/nathanluskey/ml_in_prod).

The API provided a simple way to easily switch embedding models and Spaces provides a great way to host demos.

## What are some strengths and weaknesses?
The strengths are that ML Engineers and Data Scientists can have quick access to many models. The models can be adopted from common APIs, so trying many options becomes much easier.

The weaknesses are the other half of this coin though. I actually wrote a bug in the code above because the ```xlm-roberta-base``` model will break my demo. There are still subtle differences between models that can cause bugs if not tested thoroughly. It may seem completely painless, but there's still a need to read through documentation and have a baseline understanding of the differences between models. Furthermore, with such a high level API, lots of low level control is sacrificed. 

The final consideration (neither weakness nor strength) is that HuggingFace is very new and always evolving. In the middle of making my demo, they [introduced new pricing](https://huggingface.co/blog/pricing-update). I honestly have no idea if my demo will be able to keep running for years, months, or merely weeks. There's constant improvement, new features and models, but this also comes at the cost of stability.



### Note
This post was made for my final assignment in [Carnegie Mellon's 17-645 Machine Learning in Production](https://ckaestne.github.io/seai/F2022/).