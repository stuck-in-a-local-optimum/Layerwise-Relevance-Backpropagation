# Layerwise-Relevance-Backpropagation
A research project to implement Relevance Back Propagation to explain the predictions of a hate speech detection model built on fine tuning XLM-roberta from Hugging-Face.

Layer-wise Relevance Propagation (LRP) is a technique that adds explainability to prediction of complicated deep neural networks .
LRP works by propagating the prediction (say f(x))  backwards in the neural network using specially developed local propagation rules.
In general, LRP's propagation technique is subject to a conservation property, which states that what a neuron receives must be transferred in an equivalent proportion to the bottom layer.
