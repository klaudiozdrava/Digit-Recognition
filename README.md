# Digit-Recognition

A Python project that was developed as a university assignment for the subject of Signal Processing and Voice Recognition.
The goal of this assignment was to make an ASR system that predict digits from a voice signal using Neural Network.
The steps of the algorithm are :
1) We train a simple Feed Forward Neural Network model using only Mel Spectogram as features. 
2) Seperate foreground from background information using REPET algorithm.
3) In the foreground signal,we extract digits information using sliding window technique.
4) Finally we feed our model with these digits and make predictions.

The dataset that was used for the purpose of this assigment is AudioMNIST.
