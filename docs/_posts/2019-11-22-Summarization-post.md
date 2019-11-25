---
layout: post
title: A Short Summary of Automatic Summarization
---

In the past decade, neural networks have spearheaded a renewed interest in AI due to their ability to perform quite well in tasks that involve a lot of implicit knowledge such as computer vision and NLP. Within NLP (specifically generation and understanding), automatic summarization is one of the areas where a lot of advances have been made. The purpose of this post is to provide a brief introduction (one might say, a brief *summary*) to the field of automatic text summarization. Given that the field has quite some history, we will be focusing mostly on deep learning methods.

# What is summarization

Automatic summarization is one of the most important tasks that has been explored in the fields of NLU/NLG, with interest in it for almost 20 years (Udo and Mani, 2000; 29-36). The general definition of it is largely agreed upon: **summarization** involves transforming some text into a more condensed version of the original one while remaining faithful to the original text. For instance, we can remove sentences, rephrase sentences, or merge sentences (Jing, 2002; 527-543).

There are multiple setups in which summarization occurs, such as:
- Sentence-to-sentence: We have one sentence and we try to find a summarization of it; often this involves rephrasing the sentence in a shorter manner.
- Paragraph-to-paragraph: We have a paragraph and try to "write" a shorter paragraph. This can involve either identifying the most important sentences of the paragraph, or rewriting a smaller paragraph.
- Multiple documents: This might be the summarization with largest scope since it involves taking multple documents (often with similar or repeated ideas/facts) and having a more succint summarization.


## Extractive summarization
There are two main approaches that exist to summarization: extractive and abstracive. **Extractive summarization** involves selecting the most important parts of the source text. We can think of it as having a machine read a textbook and highlight the most important passages of the book. This approach to automatic summarization has been how the problem has been approached historically (Udo and Mani, 2000; 29-36).

[HAVE SOME SMALL GRAPHIC HERE WITH AN EXAMPLE OF SOME HIGHLIGTHED TEXT]

## Abstractive summarization

**Abstractive summarization** is a setup that is less restrictive than extractive, since you are no longer limited to just using sentences in the source text. If extractive is like using a highlighter on some text, abstractive involves reading the text and taking some notes about the text.

[HAVE SOME SMALL GRAPHIC HERE WITH EXAMPLE OF NOTES] 

In some instances, allowing the model to have more control over its generated text gives it more powerful tools; for instance, it can shorten or combine sentences and it could even rephrase sentences in a clearer way. However, this freedom often makes the problem harder since it involves having to generate new sentences, adding in a more complex generative component to this problem.

# Evaluation


## Datasets
Some of the important data sets are the following
DUC


A lot of times, datasets are created by the authors of the methods by scraping the web for their desired datasets. This often leads researchers to have datasets that have readiliy available summaries, such as news sources or encyclopedia-style web pages. For instance, (Liu et al. 2018) utilized Wikipedia articles as a dataset for summarization, with the article references (and the web searches) as the input text, and the first Wikipedia paragraph as the target summary.

#### A note on datasets

An important issue that has been noted by researchers (Kryscinski et al. 2019) is that the relience on news articles for summarization stems from the following:
- They have examples of "good" summaries, namely the titles
- There are multiple sames of the same news story, allowing for more training data
- News articles have a structured setup (important information to least important)

Though this might allow the model to more easily learn news summarization, not all text that we want to summarize has said structure. Keeping this caveat in mind is necessary when using these methods since the models might not be able to properly generalize to other text sources.


## Metrics

Generative machine learning has a problem that discriminative machine learning has managed to solve more successfully: useful metrics for evaluation. For instance, if we say that a model has 95% accuracy when classifying an image as a dog or not a dog, it is clear to us what that means and if it is good or not.

Generation, however, presents a much more difficult task since we need to figure out a way of identifying if a generated data point is "good". Furthermore, the fact that there exist multiple "good" samples makes it a lot more difficult. going back to the dog example, either it is a dog or not, and one answer is correct. However, if we were trying to generate a model that can create dogs, this becomes more difficult since there are multiple "correct" dogs.

Autmatic text summarization suffers from this problem as well. Not only are there multiple correct summaries, but we also have to ensure that the metrics we choose highlight what we think is "good" of a good summary.


### Perplexity

Though not frequently used, perplexity (Brown et al.; 1992) is still used to evaluate some automatic summaries. **Perplexity** aims to measure how likely a sentence is, with a higher perplexity indicating that the sentence is less likely. This measure is mostly used for abstractive summarization since it also involves having to create new sentences.

### ROGUE(s)

One of the most widely used metrics is the Recall-Oriented Understudy for gisting Evaluation, commonly known as **ROGUE** (Lin, 2004; 74-81); this metric was elaborated especifically for evaluating summaries. In reality, ROGUE various different sub-metrics that are used, with the original paper introducing four of these: ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S. Additionally, a lot of these can be oriented towardsrecall, precision, or f-score. The specifics of these are quite different, but they all operate under the same basic logic: we compare a proposed summary sentence bar_y with a known summary hat_y and calculate how "good" of a summary hat_y is based on some similarity metric with bar_y. Another similarity between these metrics is that a higher ROGUE score indicates a "better" summary.

Note that in addition to the original ROGUE metrics, there has been a lot of subsequent work focusing on having additional ROGUE metrics (METRIC PAPER HERE).

#### ROGUE-N

*ROGUE-N* is based on the notion of *recall*, meaning that it measures how well the proposed summary can recall the appropriate elements of the reference summary. It obtains the n-grams of both summaries and obtains the ratio of the n-grams in common and the n-grams in the reference summary.

ROGUE-N also allows for multiple reference summaries by obtianing the n-grams between the proposed the summary and each of the reference summaries.

(**NICE SIMPLIFIED EQUATION OF ROGUE-N**)

(**EXAMPLE OF MEASURING ROGUE-N**)

#### ROGUE-L, ROGUE-W, and ROGUE-S

The other three ROGUE metrics measure different aspects of similarity, since ROGUE-N only focuses on recall and on number of common matches. Furthermore, these three ROGUE scores can be recall, precision, or f-score oriented.

- *ROGUE-L* looks at the longest common subsequence of words between the two summaries
- *ROGUE-W* focuses on the weighted longest common subsequence, where we give higher weight to continuous subsequences rather than subsequences with a lot of discontinuities.
- *ROGUE-S* is similar to ROGUE-N, but rather than looking at N-grams, it focuses on skip-grams.

### Fact-based metrics

Recall that in our definition of summaries, we said that summarization involves "transforming some text into a more condensed version of the original one while remaining faithful to the original text". While it is easy to see how to create a condensed text (e.g. remove ~~unnecessary~~ words from the text), remaining "faithful" to the original text can be quite hard. ROGUE aims to stay faithful to the text by ensuring that similar words are used. Yet, we often want summaries that use simpler synonyms rather than permeating the generated document with the original's superfluous and indecepherable lexicon. To that end, fact-based metrics of success have been proposed and used by researchers who want to ensure that the summary does not make any inaccurate statements or it omits any important information.

The hardest part of evaluating factual accuracy is that we would need to have an automatic way of comparing factual accuracy between two texts. The simplest way of checking for factual correctnes is by manually having humans look at the facts from the original sentence and seeing if the summary reflects the same (Cao et al. 2018). Though effective, this method seems to be hard to scale up.

Recently, there has been some work in trying to automate this process. One of the more recent works has trained a model to be able to extract facts from text (Goodrich et al. 2019). By taking this approach, they are able to extract facts from the original and the summary text, and compare these two with simple precision and recall metrics.

# Methods

For each of the main approaches within extractive and abstractive summarization, we will be focusing on explaining and understanding one model that employs said methods. Though this limits the scope of the survey, we believe that these examples highlight the main features of the approaches and that understanding these examples leads to an easier understanding of similar work.

## Background on methods

The main deep learning techniques that are used in summarization techniques stem mostly from the techniques used in NLP tasks. Among these, the most commonly used are:
- RNN's, different types of recurrent cells (e.g. GRU's and LSTM's), and bidirectional variants of these
- Attention mechanisms, often mixed with RNN's
- Transformers
- Word embeddings to obtain features from text input
- CNN's in order to have some context about each word
- Greedy and beam search in order to generate sequences of words

## Extractive Methods

Extractive summarization has had a long history of being developed, with a lot techniques outside the field of deep learning (Allahyari, et al. 2017). Within deep learning, the problem of extractive summarization is often viewed as either a supervised learning approach in which we try to classify if a sentence should be in the summary, and a reinforcement learning one where we have an agent decide if a sentence should be selected.

### Classifier approach

(Yin and Pei 2015)

### RL

(Narayan, Cohen, and Lapata 2018)

## Abstractive Methods

As mentioned in the introduction, abstractive methods aim to rephrase the source material in a shorter way. Given that we are no longer just selecting sentence sfrom the original text, a lot of non-deep learning methods are no longer applicable. Furthermore, in order to train the models, the learning setup is often different from the classifier approach. Rather than identifying which sentences to select, often the aim is to learn a language model that can predict the next word based on the context and the previous words.

### Attention-based methods

Attention models have been successfully used for datasets where there are long sequences and we want to capture information from all throughout the sequence (Cohan et al. 2018). One of the methods that has captured the attention (lol) of many is (Rush, Chopra, and Weston 2015), with it being one of the first instances of effective neural-based summarization. In this paper, they set out to teach a model how to summarize individual sentences, i.e. how to rephrase and condense a sentence. 

The model seeks out to maximize the probability of a word given the original text and the previous word (i.e. it wants to maximize the language model probabilities); consequently, during training they minimize the following negative log-likelihood equation.

(INSERT NEGATIVE LOG LIKELIHOOD EQUATION WITH A BRIEF DESRIPTION OF SOME KEY TERMS)

where X is ...., y is ...... . 

Regarding the architecture, they are using a standard encoder-decoder architecture. In the encoder part, they experiment with three different word encodings: bag-of-words, convolutional encoder, and an attention-based encoder. For the decoder, they experiment with two ways of generating word sequences: a greedy algorithm that samples the most likely word, and a beam search algorithm.

(INSERT PAPER'S PIC OF THE ARCHITECTURE, WITH BRIEF DESCRIPTION OF SOME VARIABLES)

where X is ...., y is ...... . 

The dataset that they use the Gigaword dataset (GIGAWORD REFERENCE) and they evaluate their results on a heldout subset of the Gigaword dataset as well as the DUC-2004 dataset (DUC-2004 REFERENCE). The set up with these datasets has the first sentence of a news story as the input text, and the "target" summary is the title of the news story. To evaluate their results, they evalute the ROGUE-N (N=1, N=2) and ROGUE-L scores. They also show the perplexity of their generated results to check if these sentences "make sense". All of their results show that their model outperforms all of their defined baselines (which are mostly models based on linguistics and statistics) in all metrics.

Despite the advances of this method, it is important to highlight some limitations. First, it is a sentence-to-sentence summarization rather than a document-to-paragraph one; so, in the end you would likely end up with the same number of sentences. Along with this, the small length of the source material does not accurately reflect one of the main issues that comes up in summarization: having to identify important sentences/ideas. Finally, this method does not take into account the factual accuracy of the text.


### Multi-task/ Multi-Reward

(Guo, Pasunuru, and Bansal 2018)

Intro paragraph
- problem they set out to solve
- main approach that they used

Approach
- Explanation of the method/model that they used
	- Main model that is used
	- A bit of details on the neural architecture (RNN/CNN/FFNN, attention/transformer?, encoder/decoder)
	- Learning setup (optimization algorithm, loss function)
- What makes this different from other approaches
- Include graph of setup or some equation

Experiment and Results
- Dataset that they used
- Metrics that they used (additional details if not already known)
- Graph/Table of results


Additional comments
- Limitations if any
- Links, if possiblee

### Unsupervised

(Yousefi-Azar and Hamey, 2017)


## Hybrid

In addition to purely extractive and abstractive methods, there has also been some work done in the intersection of these (Chen and Bansal 2018). The setup for these often involves using the extraction methods in order to select what sentence or paragraph is pertinent to the summary, and then using abstractive methods to compose a new text.

One paper that has gained traction in recent years is by scientists at Google Brain (Liu et al. 2018). In this project, they aim to automatically generate Wikipedia summaries at the beginning of each article. The input for the model are the article's reference sources and the Google search results when querying the article's title; and the output is the generated summary of the article. By having an extractive phase followed by an abstractive one, they manage to train a model that is able to create decent Wikipedia summaries. What distinguishes this paper from others is its use of Wikipedia as a dataset as well as its mix of extractive and abstractive methods.

The first phase of the model is the extractive one, where they look at all paragraphs of all input documents and they rank them to assess how important is each paragraph. The methods that they test for extraction are all count-based, such as using *tf-idf* (REFERENCE HERE), bi-grams, word frequencies, and similarities. 

(**INSERT SOME PIC SHOWING HOW THE EXTRACTIVE THING WORKS**)

After obtaining the ranking of the paragraphs, the abstractive part of the model uses these ordered paragraphs to generate the summary. To do so, they used a transformer architecture that does not use the encoder part, and they use the transformer architecture to predict the language model probabilities. They generate the words by using beam search of size 4 and a length penalty of 0.6, and they try to optimize for perplexity.

To evaluate their success, they used ROGUE-L F1, perplexity, and a qualitative human evaluation where they would state their preferences and evaluate the linguistic quality of the text. In additiion to showing that their model performs quite well in the measures they list, they also show that the mix of the extractive and abstractive phases significantly improves the quality of the summaries.

(**INSERT GRAPH SHOWING THAT THEIR RESULTS ARE GOOD**)

where TextRank, SumBasic, and tf-idf are different extractive methods, and T-DMCA is the transformer architecture they used for the abstrative phase.



# Closing remarks

Something about this stuff being important because of a lot of content on the web. this no longer is just a preview of the text, in some instances summaries are all eanyone will read, so in a way its "creating" new realities if not done properly.
there is a lot of room in this area for growth and contributions, not only from the point of view of new models, but also on newmetrics, problem setups, datasets and evaluation/validaito ntechniques.


# External links


# References

- (Udo and Mani 2000)
- Hahn, Udo, and Inderjeet Mani. "The challenges of automatic summarization." *Computer 33*, no. 11 (2000): 29-36.

- (Rush, Chopra, and Weston 2015)
- Rush, Alexander M., Sumit Chopra, and Jason Weston. "A Neural Attention Model for Abstractive Sentence Summarization." In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pp. 379-389. 2015.

- (Lin 2004)
- Lin, Chin-Yew. "Rouge: A package for automatic evaluation of summaries." In *Text summarization branches out*, pp. 74-81. 2004.

-(Kryscinski et al., 2019; 540-551)
- Kryscinski, Wojciech, Nitish Shirish Keskar, Bryan McCann, Caiming Xiong, and Richard Socher. "Neural Text Summarization: A Critical Evaluation." In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pp. 540-551. 2019.


- (Jing 2002)
- Jing, Hongyan. "Using hidden Markov modeling to decompose human-written summaries." *Computational linguistics 28*, no. 4 (2002): 527-543.


- (Brown et al. 1992)
- Brown, Peter F., Vincent J. Della Pietra, Robert L. Mercer, Stephen A. Della Pietra, and Jennifer C. Lai. "An estimate of an upper bound for the entropy of English." *Computational Linguistics 18*, no. 1 (1992): 31-40.


- (Goodrich et al. 2019)
- Goodrich, Ben, Vinay Rao, Peter J. Liu, and Mohammad Saleh. "Assessing the factual accuracy of generated text." In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pp. 166-175. ACM, 2019.

- (Cao et al. 2018)
- Cao, Ziqiang, Furu Wei, Wenjie Li, and Sujian Li. "Faithful to the original: Fact aware neural abstractive summarization." In Thirty-Second AAAI Conference on Artificial Intelligence. 2018.


- (Cohan et al. 2018)
- Cohan, Arman, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian. "A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents." In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, Volume 2 (Short Papers), pp. 615-621. 2018.

- (Guo, Pasunuru, and Bansal 2018)
- Guo, Han, Ramakanth Pasunuru, and Mohit Bansal. "Soft Layer-Specific Multi-Task Summarization with Entailment and Question Generation." In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics* (Volume 1: Long Papers), pp. 687-697. 2018.


- (Chen and Bansal 2018)
- Chen, Yen-Chun, and Mohit Bansal. "Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting." In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics* (Volume 1: Long Papers), pp. 675-686. 2018.


- (Yin and Pei 2015)
- Yin, Wenpeng, and Yulong Pei. "Optimizing sentence modeling and selection for document summarization." In *Twenty-Fourth International Joint Conference on Artificial Intelligence*. 2015.


- (Narayan, Cohen, and Lapata 2018)
- Narayan, Shashi, Shay B. Cohen, and Mirella Lapata. "Ranking Sentences for Extractive Summarization with Reinforcement Learning." In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, Volume 1 (Long Papers), pp. 1747-1759. 2018.


- (Yousefi-Azar and Hamey, 2017)
- Yousefi-Azar, Mahmood, and Len Hamey. "Text summarization using unsupervised deep learning." *Expert Systems with Applications 68* (2017): 93-105.


- (Allahyari, et al. 2017)
- Allahyari, Mehdi, Seyedamin Pouriyeh, Mehdi Assefi, Saeid Safaei, Elizabeth D. Trippe, Juan B. Gutierrez, and Krys Kochut. "Text summarization techniques: a brief survey." arXiv preprint arXiv:1707.02268 (2017).



- (Liu et al. 2018)
- Liu, Peter J., Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and Noam Shazeer. "Generating wikipedia by summarizing long sequences." arXiv preprint arXiv:1801.10198 (2018).


- (Kryscinski et al. 2019)
- Kryscinski, Wojciech, Nitish Shirish Keskar, Bryan McCann, Caiming Xiong, and Richard Socher. "Neural Text Summarization: A Critical Evaluation." In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 540-551. 2019.