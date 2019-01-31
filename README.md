# Sentiment Analysis - Kaggle competition “Sentiment Analysis on Movie Reviews”

### Abstract
This project presents a survey regarding sentiment analysis on the Rotten Tomatoes dataset from the Kaggle competition “Sentiment Analysis on Movie Reviews”, which was arranged between 28/2/2014 to 28/2/2015. A comparison of different machine learning algorithm is presented in addition to a to a state-of-the-art comparison. The paper presents how a logistic regression classifier is able to achieve 61.5% accuracy and outperform more than half of the Kaggle competitors.

## The Dataset
Pang and Le originally collected The Rotten Tomatoes movie review dataset [1], which this report will be focusing on. This dataset has been part of a Kaggle competition “Sentiment Analysis on Movie Reviews” that was released in 2014. The dataset has also been used in other studies [2], [3], [4]. The dataset is a corpus of movie reviews and contains a list of reviews expressed in single sentences. One training example consists of four columns; PhraseId, SentenceId , Phrase, and Sentiment. PhraseId is unique for every training example in the dataset and it exists several sub-phrases for each original sentence in the dataset. The original sentence and all sub-sentences have the same SentenceId. Phrase indicates the phrase in each example. Sentiment corresponds to the target sentiment of the phrase.

A phrase corresponds to one out of five different sentiments:
* 0 - Negative
* 1 - Somewhat negative
* 2 - Neutral
* 3 - Somewhat positive
* 4 - Positive

The dataset contains 8529 reviews that have been split into 156060 subphrases, each with its own sentiment target. 51% of the phrases are labelled as neutral, 21% as somewhat positive, 17.5% as somewhat negative, 6% as positive, and 4.5% as negative. A diagram showing the sentiment distribution is shown below in Figure 1.

![logo](./Images/image1.png?raw=true)

**Figure 1**. Graph showing sentiment distribution in the training data.

## Experiments and Analysis

### Pre-processing

Several basic state-of-the-art techniques were implemented and analysed to minimize the error rate. Tokenization and converting the tokens to lowercase words were initially performed in addition to removing stop words. After analysing this basic approach, it became clear that the removal of stop words resulted in an increased error rate. This might be a result of the fact that several phrases in the dataset only consist of stop words. These phrases might, therefore, be wrongly classified and lead to an increased error rate.

A new experiment was tested were stop words were removed in phrases that did not only contain stop words. This approach also resulted in a slightly worse performance and stop words was therefore included as input features. Further experiments were conducted by analysis the removal of words that starts with numbers. This approach resulted in an increased error rate. Removal of words starting or ending with a special character was tested and resulted in a slightly improved performance. Stemming was then tested, which resulted in a performance boost in addition to reducing the total word count of the generated corpus. Lemmatization was tested but did not affect the performance in any noticeable way. Removal of single characters also increased the accuracy, probably due to its high probability of being punctuations. Several experiments with n-grams were conducted, which resulted in an increased training and prediction time. The performance boost of adding bigrams was estimated to around 0.3%. This slight performance boost might have occurred due to proper negation capturing.

To further capture negations, "\_NEG" was appended to words that appeared in a negation context, which was hardcoded as a trigger for the word “not”. This is a classical approach to tackle this issue [5]. Additionally, “\_VERY” was appended to sentences that contained the word “very”. Both these approaches resulted in a slight performance improvement. POS-tagging was also tested by concatenating each word with its calculated tag. This approach resulted in a noticeable worse performance. TFIDF was calculated which was then used to remove outliers, such as words that appear frequently and rarely. This approach did not improve the performance either. Creating a custom stop word list by analysing the results from the term frequency calculation was tested, but did not improve the accuracy. All the experiments above were tested using a subset of the training data (10000 examples) and using 3-fold cross-validation on unseen data to provide the accuracy estimates.

A word cloud was created to visualize term frequencies words (stop words are not included), which is shown below in Figure 2.

![logo](./Images/image2.png?raw=true)

**Figure 2**. Word cloud of common words in the dataset.

### Classifiers

A dummy classifier was chosen for this study and achieved an accuracy of 0.34%. Several classification algorithms were tested after the optimization of the preprocessing step. The comparison involved training each classifier using 10000 examples and estimating the accuracy using 3-fold cross-validation. Bigrams were not included in this estimation due to time complexity. Most of the chosen approaches are known to perform well in text analytics and was selected due to their popularity in academia. The implemented classifiers originate from sci-kit learn and were converted to NLTK classifiers using nltk.classify.scikitlearn. Each classifier was manually tuned for optimal performance. The selected classifiers and their estimated accuracies are listed below in Figure 3.



| Classifier   | Accuracy        |
|-             |-                |
| Baseline     | 0.3419          |
| Support Vector Machine (SVM)     | 0.5484        |
| Multinomial Naive Bayes     | 0.5594   |
| Naive Bayes Bernoulli     | 0.5636   |
| Logistic Regression     | 0.5743   |
| Gradient Boosting     | 0.5618   |
| Decision Tree     | 0.5185   |
| Stochastic Gradient Descent     | 0.5545   |
| Random Forest     | 0.5398   |
| Voting Classifier (Includes all of the above)     | 0.5681   |
| Multi Layer Perceptron (MLP)     |  0.5412   |

**Figure 3**. Word cloud of common words in the dataset.

## Performance and Result

The final implementation of the pre-processing step includes tokenization,lowercase filtering, removal of single characters, removal of words starting or ending with special characters, stemming, adding bigrams, negation capturing, and very capturing. The code-flow of the pre-processing step is showed below in Figure 4.

```
tokenize(sentence)
lowercase(sentence)
remove_single_character(sentence)
remove_start_end_exceptions(sentence)
stem(sentence)
capture_negation(sentence)
capture_very(sentence)
bigrams(sentence)
```

**Figure 4**. Pre-processing steps.

The logistic regression was chosen as the final classifier due to its superior performance in the experimental study. The classifier was then trained on the entire training set. Input words were not chosen as part of the feature set for the classifier unless the classifier had trained on the words before. The classifier then produced the final predictions using the provided test set. The result was then uploaded to the Kaggle website for validation, which returned a final estimated accuracy of 61.5%.

## Conclusion and Review

Most of the basic approaches known from the state of the art were tested during this study and some with little success – lemmatization, and POS tagging did not improve the performance, rather the opposite. Bigrams only improved the performance by a very small margin. There are limited papers providing specific accuracy estimates and directional indications, which makes it difficult to conduct what should have been done differently. Some related work has included the dataset, but not estimated accuracies specifically. A completely different approach, for example a recurrent neural network, might have resulted in a better performance due to the limitation of the bag-of-words approach [7].

This study shows that well-known approaches for tackling sentiment analysis might not improve the accuracy by a large margin and might even increase the prediction error in some cases. However, the training set used in this study might be constructed in a questionable way. One example is that the phrase “tries for more .”, which is labelled as somewhat negative, while “tries for more” is labelled as neutral. The only difference between these two phrases is the full stop at the end. Some of the examples in the dataset may not feel strictly intuitive and it might be the reasoning why the removal of stop words resulted in a worse performance.

This prediction model would be ranked as number 357 by comparing the result to the public leaderboard on the Kaggle website. This study shows that by using known state-of-the-art techniques and an optimized logistic regression classifier, it is possible to achieve 61.5% accuracy and outperform more than half of the participants shown in the Kaggle leaderboard.

## References
1. Pang. B and Lee. L, (Jun, 2005), Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales, ACL, arXiv preprint arXiv:cs/0506075.
2. Jean Y. Wu, Yuanyuan Pao, “Predicting Sentiment from Rotten Tomatoes Movie Reviews”, Internet: https://nlp.stanford.edu/courses/cs224n/2012/reports/WuJean_PaoYuanyuan_224nReport.pdf, [24. Feb, 2018].
3. Miyato. T, Dai. A, Goodfellow. I, (Nov, 2016), Adversarial Training Methods For Semi-Supervised Text Classification. arXiv:1605.07725.
4. Yu. A, Lee. H, Le. Q, (Apr, 2017), Learning to Skim Text. arXiv preprint arXiv:1704.06877
5. Nakov. P, (Oct, 2017), Semantic Sentiment Analysis of Twitter Data. arXiv preprint arXiv:1710.01492.
6. Kharde. V, Sonawane. S, (April, 2016), Sentiment Analysis of Twitter Data: A Survey of Techniques. Volume 139 – No.11 arXiv preprint arXiv:1601.06971.
7. Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts: ‘Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank’, in: Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pp. 1631–1642, Seattle, WA, USA, 2013.
