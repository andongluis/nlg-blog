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


## NLP

Natural Language Processing (NLP for short) is a broad term used to describe Machine Learning/Artificial Intelligence techniques for understanding language. The name of the area is a bit of a misnomer since some people recognize three broad areas that NLP has:
- **Natural Language Processing (NLP)**: How do we process "human-understandable" language into "machine-understandable" input
- **Natural Language Generation (NLU)**: How can we use machines to generate sensical speech and text
- **Natural Language Understanding (NLG)**: How can we make machine understand what we mean when we communicate
Note that most problem end up having overlaps between the three areas, with NLP being quite necessary for the other two. For instance, it is hard for a machine to understand language if it is not able to first process it.

### Language Models

In order for us to understand **NLG**, we often resort to a mathematical understanding of how language can be generated. Specifically, a **language model** is a thingy that has the probability of the next word dependent on all future words. This makes sense since what I say in the future will likely depend on what I've already said.

Also mention backwards probabilities as important


### Word Representations

**Word representations** are things (often vectors) that we use to represent words in a machine-understandable way. They are especially useful for ML algorithms since the vector representation allows for statistical methods to be used, especially optimization-based algorithms.


### Distributed Word Representations

Among the many different kinds of word representations, the one that has gained the most traction over the past few years is **distributed word representations**. These use condense feature vectors in latent spaces that are not immediately human-understandable. Deep Learning is often used with these.


## Deep Learning

So deep I can't even see you


### Basics (?)


### Residual Networks (?)

Residual networks help "refresh" the network's memory of the original input. It can also be used to "short-circuit" connections in case that you don't actually need that many layers to learn something


### Convolutional Neural Networks

![convnet image]({{site.baseurl}}/images/convnet-kdnuggets.png)

Go over idea that stuff close together matters.
Fixed window is an issue, can be solved with recurrent

### Recurrent Neural Networks

Some stuff on recurrent nets, focus on the intuition that the previous stuff depends on previous stuff
Mention how this related to language model idea

### LSTM Cells

Helps maintain memory and helps more with long-term dependencies and training

### Bidirectional RNNs

Some stuff is useful to understand it forwards and backwards to understand an entire sentence. Not used for prediction (since you are assuming you "know the future"), but can be used to understand an entire sequence.


### Attention Mechanisms

Helps identify what part of the sequence is important. not all parts of the sequence are made equal
 There are a lot of different kinds of attention mechanisms


### Transformers

This is used as a way to understand sequential data without RNN's and by only using attention mechanisms