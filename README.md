# Final Project for Statistical Learning in Msc. Data Science, Sapienza University of Rome.

This project explores various techniques for machine unlearning, which refers to the process of removing the influence of a subset of training data from a machine learning model. The goal is to align the performance of the unlearned model with a model retrained from scratch on the remaining data, without compromising the model's performance on the retained data.
The methods evaluated in this study include:

-Finetuning: A baseline approach that fine-tunes the model on the retained dataset.

-Random Perturbation Unlearning: A simple method that adds Gaussian noise to the model parameters to help forget the forget dataset.

-Stochastic Teacher Network: A two-step approach that first uses a randomly initialized network to erase the influence of the forget dataset, and then reconstructs the model to maintain performance on the retained dataset.

-Fisher Masking: A technique that uses the Fisher Information Matrix to identify and mask the parameters most important for the forget dataset.

-Selective Synaptic Dampening: An extension of Fisher Masking that selectively dampens the important parameters for the forget dataset instead of masking them

The performance of these techniques is assessed on metrics such as retained set accuracy, forgotten set accuracy, and computational complexity. The project provides insights into the strengths and limitations of each method, aiming to contribute to the growing field of machine unlearning.
