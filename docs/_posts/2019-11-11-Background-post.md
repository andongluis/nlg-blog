---
layout: post
title: NLG Background
---

These blog posts will focus on the problem of Natural Language Generation (NLG). Over various blog posts, we will explore key models that have significantly advanced the field, along with an exploration of some important problems that are trying t obe solved in NLG.

This blog post will focus on the Natural Language Processing (NLP) and Deep Learning (DL) background required to understand the rest of the blog posts. However, due to the extensive nature of these topics, we will assume some background knowledge in order for us to focus on the NLG-specific aspects of DL. Specifically, we will assume that readers will be familiar with
- What it means to train a DL model
- What regularization means
- Overfitting and underfitting
- Neural network basics: feed-forward neural networks, activation functions, backpropagation basics


# NLP

Natural Language Processing (NLP for short) is a broad term used to describe Machine Learning/Artificial Intelligence techniques for understanding language. The name of the area is a bit of a misnomer since some people recognize three broad areas that NLP has:
- **Natural Language Processing (NLP)**: How do we process "human-understandable" language into "machine-understandable" input
- **Natural Language Generation (NLU)**: How can we use machines to generate sensical speech and text
- **Natural Language Understanding (NLG)**: How can we make machine understand what we mean when we communicate


**GENERIC PIC OF NLP OR NLG**


Note that most problem end up having overlaps between the three areas, with NLP being quite necessary for the other two. For instance, it is hard for a machine to understand language if it is not able to first process it.

## Language Models

In order for us to understand **NLG**, we often resort to a mathematical understanding of how language can be generated. Specifically, a **language model** is a way of modeling probability of a word or sequence of words. For instance, if we want to model how likely the phrase "did nothing wrong", we would write down the probability as 

$$p(w_1=did, w_2=nothing, w_3=wrong)$$

Often, we want to model word and sentence probability with added conditions. For instance, if we wanted get the likely of the phrase "Thanos did nothing wrong" given that the previous word was "Thanos", then that probability would be 

$$p(w_1=did, w_2=nothing, w_3=wrong | w_0=Thanos)$$
[**NEED REFERENCE FOR LM**]

### N-grams

Given that sentences can be quite long (Gabriel García Márquez famously wrote a two-page long sentence in *One Hundred Years of Solitude*), it is often useful to just look at a subset of that sequence. To that end, we use n-grams which basically ask the question: given the previous *n-1* words, what is the probability of the *n*-th word? In probability terms, this looks like

$$p(w_n | w_1, w_2, ..., w_{n-1}) $$
[**NEED REFERENCE FOR N GRAM**]

### Skip-grams

N-grams are limited by the sequential nature of the n-gram; you are using as context the *n-1* words before the *n*-th word and only those. A more flexible approach has been **skip-grams** which still will use some *n* number of words as context, but it allows you to *skip* over some words. For instance, when predicting the word following the sentence "The dog happily ate the " and we choose to have *n=2*, we might choose the words ["dog", "ate"] rather than having to focus solely on the words ["ate", "the"].

[**PIC ON SKIPGRAMS**]
[**NEED REFERENCE FOR SKIP GRAM**]

### Forward and Backward probabilities

So far, we have seen probabilities of the next word given previous words. These types of probabilities are called **forward probabilities**.

$$p(w_n | w_1, w_2, ..., w_{n-1}) $$

However, we can also look at **backward probabilities**, meaning the probability of a past word given the current words.

$$p(w_1 | w_2, w_3, ..., w_{n}) $$

Though not as useful for predicting future words, they are still useful for understanding sequences and obtaining features for the context surrounding words.

[**REFERENCE FOR THIS, MAYBE IN ELMO**]

## Word Representations

**Word representations** are ways of representing words in a machine-understandable way; the most common way of doing so is by representing a word as a vector. They are especially useful for ML algorithms since the vector representation allows for statistical methods to be used, especially optimization-based algorithms.

As an example, suppose we wanted to represent the sentence "The dog is the goodest of boys". One way we could do it is by using one-hot vector encodings. Then, th

**PIC WITH A VECTOR FOR EACH WORD**

Here, we define a dimension for each word and the vector for a word will have a 1 if that word is present, 0 otherwise. We could also represent an entire sentence with this word space. For instance, the sentences "The dog is the goodest of boys" will be 

**PIC WITH A VECTOR OF ONES**

The sentence "The dog is good" would look like

**PIC WITH CORRESPONDING VECTOR**

Note that since we did not include "good" in our original dictionary, this does not appear in our vector.

### Distributed Word Representations

There are multiple types of word representations, such as based on clustering, one-hot encodings, and based on co-occurrences. However, as we saw in the previous example, a big problem is the sparsity of the space (meaning that we have too many dimensions for too few data points) and that we might not handle unseen words well (i.e. we are unable to generalize).

Among the many different kinds of word representations, the one that has gained the most traction over the past few years is **distributed word representations** (also called **word embeddings**. These use condense feature vectors in latent spaces that are not immediately human-understandable. This means that we can generalize to unseen words and the problem of sparsity is handled better. Deep Learning is often used with these.

For instance, we could run a neural network on the word "dog" and get a latent space encoding.

**PIC OF WORD OVER A NEURAL NETWORK THAT GIVES SOME ENCODING**

Though we might not be able to easily understand what these dimensions mean in terms of human terms, they are often more useful for machine learning algorithms.

# Deep Learning

Within Machine Learning (ML), the area that has spearheaded a lot of the performance gains has been Deep Learning (DL). DL is the area of ML that deals with neural network-based models.


## Basics (?)


## Residual Networks (?)


Residual networks have been used quite widely for some time now [**INSERT REFERENCE HERE**] with a lot of recent architectures (such as ELMo and BERT) using these and with most state-of-the-art using these by default.

The main idea is that we have some "skip-ahead" connections in the neural architecture.

**INSERT PIC OF SOME RESIDUAL CONNECTION HERE**

There are two main ideas that motivate residual connections:
- "Short-circuit" connections: A key hyperparameter in neural architectures is the number of layers, with the trend being that more is better. However, for some instances it is not necessary to have that many layers. To alleviate some of the hyperparameter tuning work, residual connections can help give the network the ability to decide to use all layers or to ignore some of them.
- "Refresh memory": The other big idea is that by having some connections that skip layers, we can give the deeper layers of the network a refresher on what the original input looked like. That way, the network can used the latent features along with the original ones.


## Convolutional Neural Networks

Arguably, the first networks that really impressed the world [**REFERENCE HERE FOR SOME IMAGENET THING**] are convolutional neural networks (CNN). The distinguishing feature of CNN's are convolutional layers, which take into account an input's neighboring values. In other words, rather than looking at each component of the input vector as "independent", they look at each component's neighborhood.

**INSERT PIC OF A CONVOLUTION HERE**

So, a CNN ends up with convolutional layers at the beginning and then some "regular" network connections at the end

![convnet image]({{site.baseurl}}/images/convnet-kdnuggets.png)

Though CNN's are often associated with image inputs, they can also be used in sequential data. For instance, we could look at the window of 2 words before and after our current word.

**INSERT PIC OF CNN WITH WORDS**


## Recurrent Neural Networks

Recurrent Neural Networks (RNN's) deal with an important limitation of CNN's: their fixed-size window. CNN's convolutions have a fixed window that they deal with and they have a fixed sized input that they can deal with. This is problematic for sequences such as sentences since often vary in length.

RNN's can deal with this by introducing a special layer called the Recurrent layer. These use cells that take as input a part of the sequence and the output of the cell with the previous input

**INSERT PIC OF RECURRENT CELL**

Given this set up, it is easy to see how these could be used for language modeling since they can (theoretically) take into account as many previous words as possible.

##$ LSTM Cells

Normal RNN's face various limitations, the most glaring being that it might be difficult to train them due to exploding/vanishing gradients [**LINK FOR GRADIENTS**] and they can have trouble remembering long sequences.

Rather than using RNN's, what people end up using is often Long Short-Term Memory Cells (LSTM Cells). The details of LSTM cells are quite intricate, yet the intuition behind these is remarkably simple: LSTM's can have longer term memory by choosing waht to remember and what to forget.

[**PIC OF LSTM**]


### Bidirectional RNNs


RNN's are usually used for future prediction, meaning that you follow the "forward" probability Language Model. However, they can also be used to model the backward probabilities.

The rationale behind this seems a bit unintuitive: why would you want to predict what has already happened? And often, you don't actually want to predict what has happened (and you can't use them to predict the future since that is not what you're learning). The main reason why you would use them is for understanding the entire sequence rather than trying to predict future incidents.

This reasoning motivates Bidirectional RNN's, which use both forward and backward Language Models.

[**PIC OF BI-RNN**]


## Attention Mechanisms

Helps identify what part of the sequence is important. not all parts of the sequence are made equal
 There are a lot of different kinds of attention mechanisms


### Transformers

This is used as a way to understand sequential data without RNN's and by only using attention mechanisms