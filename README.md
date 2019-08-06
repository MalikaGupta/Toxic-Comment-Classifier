# Toxic-Comment-Classifier
Classification of any live comment or a database of comments into 6 toxicity categories (toxic, severe toxic, threat, 
insult, obscene, identity hate) and predicting the probability percentage using Neural Networks and Word2Vec.

In this project, I have used Wikipedia comment dataset availabe on kaggle.

The training data consists of comments and the probability(0 or 1) of each comment to be toxic,severely toxic, threat, insult,obscene 
and identity hate.
Since the training dataset is labelled, we use supervised learning method.

First the training and the test dataset is preprocessed,for example, empty rows are discarded.
With the use of Word2Vec, comments in the training dataset are converted to feature vectors and the trained Word2Vc model is
saved using joblib.

Then these feaure vectors are used to train 6 MultiLayer Perceptron Models, one for each category.
After the MLP Classifier model is trained on the dataset, we save this model using joblib.

With the use of FLASK API, we take input from the user in the form of two comments.
Two comments are taken as live input so that the user can compare.
The saved Word2Vec model is used to convert these comments to vectors.
Then the probability percentage for each category is calculated by using the output of word2vec model as input for each of the six MLP models.
