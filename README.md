## Introduction

This repository describes our work at the SemEval 2016 Task : Twitter sentiment analysis.

http://alt.qcri.org/semeval2016/task4/

## Description of the project
Task 4 is divided into 5 subtasks :
Subtask A: Multi class classification problem where the polarity of a given tweet should be correctly classified over 3 different classes: Positive - Negative - Neutral.

• Subtask B: Binary class classification problem. This time, given a tweet known to be around a given topic, we have to determine whether the tweet expresses a positive or negative opinion over the topic.

• Subtask C: Classification problem over an ordinal scale. Given a tweet around a given topic we need to determine where the polarity of the sentiment expressed in the tweet falls on a five point scale. (5 being a very positive sentiment and 0 being a very negative sentiment).

• Subtask D: Binary quantification problem. Given tweets knows to be around a certain topic, we need to determine the distribution of the tweets over the negative and positive classes.

• Subtask E: Multi class quantification problem. Similarly to Subtask D, give a set of tweets revolving around a topic, we need to determine the distribution of the tweets over a 5 points scale ( Same scale as in Subtask C).

## Results

• For subtask A we obtained an F1-measure of 0.550 and placed 20/35

• For subtask B we obtained a recall of 0.787 and placed 3/20

• For subtask C we obtained a Mean-Error of 0.910 and placed 9/12

• For subtask D we obtained a KLD of 0.07 and placed 6/15

• For subtask E we obtained an EMD of 0.365 and placed 7/11


