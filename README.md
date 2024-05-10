# Echonet

This is the repository for our project for CSCI 1470. The authors are Alvaro Franco, Aidin Moldosanov, and Santiago Enriquez.

We used a database from [Stanford's Center for Artificial Intelligence in Medicine and Imaging](https://echonet.github.io/dynamic/index.html), doing a similar approach to a published [paper](https://www.nature.com/articles/s41586-020-2145-8).
This repository contains the pre-trained weights for our trained model. 

To test the model's performance, you can call the function as `python main.py --unet load`. You would need to change the data path in the `preprocess.py` file, line 94.
