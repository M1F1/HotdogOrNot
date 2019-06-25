# HotdogOrNot

Application HotdogOrNot classify images into two groups: hot-dogs and not hot-dogs.
I used Convolutional neural network with: 3 conv2d,  dropouts, maxpooling and two dense layers.
CNN was trained and evaluate on google colab, details are in Michal_Filek_project_3_ADL.ipynb
Dataset was download from: https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog/data.
App is hosted on AWS EC2, on an virtual ubuntu instance.
Model is served via restfull api on flask server.
Adress for testing: http://52.14.217.174/
