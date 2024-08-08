# Final Project for Statistical Learning in Msc. Data Science, Sapienza University of Rome.

This project explores various techniques for machine unlearning, which refers to the process of removing the influence of a subset of training data from a machine learning model. The goal is to align the performance of the unlearned model with a model retrained from scratch on the remaining data, without compromising the model's performance on the retained data.
The methods evaluated in this study include:

-Finetuning

-Random Perturbation Unlearning

-Stochastic Teacher Network

-Fisher Masking

-Selective Synaptic Dampening

The performance of these techniques is assessed on metrics such as retained set accuracy, forgotten set accuracy, and computational complexity. The project provides insights into the strengths and limitations of each method, aiming to contribute to the growing field of machine unlearning.
