# Layerwise-Relevance-Backpropagation
A research project to implement Relevance Back Propagation to explain the predictions of a hate speech detection model built on fine tuning XLM-roberta from Hugging-Face.

Layer-wise Relevance Propagation (LRP) is a technique that adds explainability to prediction of complicated deep neural networks .
LRP works by propagating the prediction (say f(x))  backwards in the neural network using specially developed local propagation rules.
In general, LRP's propagation technique is subject to a conservation property, which states that what a neuron receives must be transferred in an equivalent proportion to the bottom layer.

<img width="615" alt="Screenshot 2022-06-29 at 17 36 35" src="https://user-images.githubusercontent.com/55681180/176432340-03435267-cedc-4f66-a664-c53a8233f8ab.png">


Let j and k represent neurons in two successive layers, say layer ‘p’ and ‘q’ in the above neural network. The rule  to back- propagate relevance scores (Rk)k from layer ‘q’ to neurons ‘j’ in the previous layer ‘q’  is the following:
<img width="282" alt="Screenshot 2022-06-29 at 17 37 42" src="https://user-images.githubusercontent.com/55681180/176432504-fdb5b93a-697c-4660-bd71-d1b428dd0ff7.png">
Note: Here the index of upper summation, i.e., “j”  represents neurons of previous layer  “p” where relevance had to reach from neuron ‘k” of layer “q” by back propagating.

And  zjk represents the extent to which neuron j in the layer ‘n’ has contributed to the R_k, the relevance of neuron k from the layer ‘q’. The conservation property was maintained by the denominator term. Once the input features are reached, the propagation procedure ends.

LRP can be used for a variety of purposes, some of which are as follows:
Let’s say our network predicts a cancer diagnosis based on a mammogram (a breast tissue image), the explanation provided by LRP would be a map showing which pixels in the original image contribute to the diagnosis and to what amount. Because this approach does not interfere with network training, it can be used on already trained classifiers.
XML methods are especially useful in safety-critical domains where practitioners must know exactly what the network is paying attention to.  Other use-cases include network (mis)behavior diagnostics, scientific discovery, and network architectural improvement.
