# Echonet

This is the repository for our project for CSCI 1470. The authors are Alvaro Franco, Aidin Moldosanov, and Santiago Enriquez.

We used a database from [Stanford's Center for Artificial Intelligence in Medicine and Imaging](https://echonet.github.io/dynamic/index.html), doing a similar approach to a published [paper](https://www.nature.com/articles/s41586-020-2145-8).
This repository contains our proposed architecture for segmenting the Left Ventricle from echocardiography videos and calculating the ejection fraction. We include our pre-trained weights, from a random sampling of the aforementioned database. Those weigths are located in `models\Unet`.

Our architecture is a U-Net, commonly used in scarce medical databases. We have also included our tries to implement a different neural network to predict the frames inside the videos that would  better ilustrate the EDV and ESV. This wasn't properly working and might be revised in the future. 

To test the model's performance, you can call the function as `python main.py --unet load`. You would need to change the data path in the `preprocess.py` file, line 94.
