# Emojify
To use word vector representations of sequence models to build an Emojifier


Have you ever wanted to make your text messages more expressive? Your emojifier app will help you do that. So rather than writing "Congratulations on the promotion! Lets get coffee and talk. Love you!" the emojifier can automatically turn this into "Congratulations on the promotion! 👍 Lets get coffee and talk. ☕️ Love you! ❤️"

You will implement a model which inputs a sentence (such as "Let's go see the baseball game tonight!") and finds the most appropriate emoji to be used with this sentence (⚾️). In many emoji interfaces, you need to remember that ❤️ is the "heart" symbol rather than the "love" symbol. But using word vectors, you'll see that even if your training set explicitly relates only a few words to a particular emoji, your algorithm will be able to generalize and associate words in the test set to the same emoji even if those words don't even appear in the training set. This allows you to build an accurate classifier mapping from sentences to emojis, even using a small training set.

In this exercise, you'll start with a baseline model (Emojifier-V1) using word embeddings, then build a more sophisticated model (Emojifier-V2) that further incorporates an LSTM.

Lets get started! Run the following cell to load the package you are going to use. (refer emoji.py)

1 - Baseline model: Emojifier-V1
1.1 - Dataset EMOJISET
Let's start by building a simple baseline classifier.

You have a tiny dataset (X, Y) where:

X contains 127 sentences (strings)
Y contains a integer label between 0 and 4 corresponding to an emoji for each sentence (refer image)

Figure 1: EMOJISET - a classification problem with 5 classes. A few examples of sentences are given here.

Let's load the dataset using the code below. We split the dataset between training (127 examples) and testing (56 examples).
(refer emoji.py)

Run the following cell to print sentences from X_train and corresponding labels from Y_train. Change index to see different examples. Because of the font the iPython notebook uses, the heart emoji may be colored black rather than red. (refer emoji.py)

I am proud of your achievements 😄

1.2 - Overview of the Emojifier-V1
In this part, you are going to implement a baseline model called "Emojifier-v1". (refer image)

Figure 2: Baseline model (Emojifier-V1).

To get our labels into a format suitable for training a softmax classifier, lets convert  YY  from its current shape current shape  (m,1)(m,1)  into a "one-hot representation"  (m,5)(m,5) , where each row is a one-hot vector giving the label of one example, You can do so using this next code snipper. Here, Y_oh stands for "Y-one-hot" in the variable names Y_oh_train and Y_oh_test: (refer emoji.py)

Let's see what convert_to_one_hot() did. Feel free to change index to print out different values. (refer emoji.py)

All the data is now ready to be fed into the Emojify-V1 model. Let's implement the model!

1.3 - Implementing Emojifier-V1
As shown in Figure (2), the first step is to convert an input sentence into the word vector representation, which then get averaged together. Similar to the previous exercise, we will use pretrained 50-dimensional GloVe embeddings. Run the following cell to load the word_to_vec_map, which contains all the vector representations. (refer emoji.py)

You've loaded:

word_to_index: dictionary mapping from words to their indices in the vocabulary (400,001 words, with the valid indices ranging from 0 to 400,000)
index_to_word: dictionary mapping from indices to their corresponding words in the vocabulary
word_to_vec_map: dictionary mapping words to their GloVe vector representation.
Run the following cell to check if it works. (refer emoji.py)

Exercise: Implement sentence_to_avg(). You will need to carry out two steps:

Convert every sentence to lower-case, then split the sentence into a list of words. X.lower() and X.split() might be useful.
For each word in the sentence, access its GloVe representation. Then, average all these values. (refer emoji.py)

Expected Output:

avg=	[-0.008005 0.56370833 -0.50427333 0.258865 0.55131103 0.03104983 -0.21013718 0.16893933 -0.09590267 0.141784 -0.15708967 0.18525867 0.6495785 0.38371117 0.21102167 0.11301667 0.02613967 0.26037767 0.05820667 -0.01578167 -0.12078833 -0.02471267 0.4128455 0.5152061 0.38756167 -0.898661 -0.535145 0.33501167 0.68806933 -0.2156265 1.797155 0.10476933 -0.36775333 0.750785 0.10282583 0.348925 -0.27262833 0.66768 -0.10706167 -0.283635 0.59580117 0.28747333 -0.3366635 0.23393817 0.34349183 0.178405 0.1166155 -0.076433 0.1445417 0.09808667]


Model
You now have all the pieces to finish implementing the model() function. After using sentence_to_avg() you need to pass the average through forward propagation, compute the cost, and then backpropagate to update the softmax's parameters.

Exercise: Implement the model() function described in Figure (2). Assuming here that  YohYoh  ("Y one hot") is the one-hot encoding of the output labels, the equations you need to implement in the forward pass and to compute the cross-entropy cost are:
z(i)=W.avg(i)+b
 
a(i)=softmax(z(i))

L(i)=−∑k=0ny−1Yohk(i)∗log(ak(i))
 
It is possible to come up with a more efficient vectorized implementation. But since we are using a for-loop to convert the sentences one at a time into the avg^{(i)} representation anyway, let's not bother this time.

We provided you a function softmax().  (refer emoji.py)

Run the next cell to train your model and learn the softmax parameters (W,b).  (refer emoji.py)

Expected Output (on a subset of iterations):

Epoch: 0	cost = 1.95204988128	Accuracy: 0.348484848485
Epoch: 100	cost = 0.0797181872601	Accuracy: 0.931818181818
Epoch: 200	cost = 0.0445636924368	Accuracy: 0.954545454545
Epoch: 300	cost = 0.0343226737879	Accuracy: 0.969696969697
Great! Your model has pretty high accuracy on the training set. Lets now see how it does on the test set.

1.4 - Examining test set performance (refer emoji.py)

Expected Output:

Train set accuracy	97.7
Test set accuracy	85.7
Random guessing would have had 20% accuracy given that there are 5 classes. This is pretty good performance after training on only 127 examples.

In the training set, the algorithm saw the sentence "I love you" with the label ❤️. You can check however that the word "adore" does not appear in the training set. Nonetheless, lets see what happens if you write "I adore you."  (refer emoji.py)

Accuracy: 0.833333333333

i adore you ❤️
i love you ❤️
funny lol 😄
lets play with a ball ⚾
food is ready 🍴
not feeling happy 😄
Amazing! Because adore has a similar embedding as love, the algorithm has generalized correctly even to a word it has never seen before. Words such as heart, dear, beloved or adore have embedding vectors similar to love, and so might work too---feel free to modify the inputs above and try out a variety of input sentences. How well does it work?

Note though that it doesn't get "not feeling happy" correct. This algorithm ignores word ordering, so is not good at understanding phrases like "not happy."

Printing the confusion matrix can also help understand which classes are more difficult for your model. A confusion matrix shows how often an example whose label is one class ("actual" class) is mislabeled by the algorithm with a different class ("predicted" class). (refer emoji.py)

(56,)
           ❤️    ⚾    😄    😞   🍴
Predicted  0.0  1.0  2.0  3.0  4.0  All
Actual                                 
0            6    0    0    1    0    7
1            0    8    0    0    0    8
2            2    0   16    0    0   18
3            1    1    2   12    0   16
4            0    0    1    0    6    7
All          9    9   19   13    6   56
What we should remember from this part:

Even with a 127 training examples, you can get a reasonably good model for Emojifying. This is due to the generalization power word vectors gives you.
Emojify-V1 will perform poorly on sentences such as "This movie is not good and not enjoyable" because it doesn't understand combinations of words--it just averages all the words' embedding vectors together, without paying attention to the ordering of words. we will build a better algorithm in the next part.

2 - Emojifier-V2: Using LSTMs in Keras:
Let's build an LSTM model that takes as input word sequences. This model will be able to take word ordering into account. Emojifier-V2 will continue to use pre-trained word embeddings to represent words, but will feed them into an LSTM, whose job it is to predict the most appropriate emoji.

Run the following cell to load the Keras packages. (refer emoji.py)

2.1 - Overview of the model
Here is the Emojifier-v2 you will implement: (refer image)

Figure 3: Emojifier-V2. A 2-layer LSTM sequence classifier.

2.2 Keras and mini-batching
In this exercise, we want to train Keras using mini-batches. However, most deep learning frameworks require that all sequences in the same mini-batch have the same length. This is what allows vectorization to work: If you had a 3-word sentence and a 4-word sentence, then the computations needed for them are different (one takes 3 steps of an LSTM, one takes 4 steps) so it's just not possible to do them both at the same time.

The common solution to this is to use padding. Specifically, set a maximum sequence length, and pad all sequences to the same length. For example, of the maximum sequence length is 20, we could pad every sentence with "0"s so that each input sentence is of length 20. Thus, a sentence "i love you" would be represented as  (ei,elove,eyou,0⃗ ,0⃗ ,…,0⃗ )(ei,elove,eyou,0→,0→,…,0→) . In this example, any sentences longer than 20 words would have to be truncated. One simple way to choose the maximum sequence length is to just pick the length of the longest sentence in the training set.


2.3 - The Embedding layer
In Keras, the embedding matrix is represented as a "layer", and maps positive integers (indices corresponding to words) into dense vectors of fixed size (the embedding vectors). It can be trained or initialized with a pretrained embedding. In this part, you will learn how to create an Embedding() layer in Keras, initialize it with the GloVe 50-dimensional vectors loaded earlier in the notebook. Because our training set is quite small, we will not update the word embeddings but will instead leave their values fixed. But in the code below, we'll show you how Keras allows you to either train or leave fixed this layer.

The Embedding() layer takes an integer matrix of size (batch size, max input length) as input. This corresponds to sentences converted into lists of indices (integers), as shown in the figure below.  (refer image)

Figure 4: Embedding layer. This example shows the propagation of two examples through the embedding layer. Both have been zero-padded to a length of max_len=5. The final dimension of the representation is  (2,max_len,50) because the word embeddings we are using are 50 dimensional.


The largest integer (i.e. word index) in the input should be no larger than the vocabulary size. The layer outputs an array of shape (batch size, max input length, dimension of word vectors).

The first step is to convert all your training sentences into lists of indices, and then zero-pad all these lists so that their length is the length of the longest sentence.

Exercise: Implement the function below to convert X (array of sentences as strings) into an array of indices corresponding to words in the sentences. The output shape should be such that it can be given to Embedding() (described in Figure 4).
(refer emoji.py)

Run the following cell to check what sentences_to_indices() does, and check your results. (refer emoji.py)

Expected Output:

X1 =	['funny lol' 'lets play football' 'food is ready for you']
X1_indices =	[[ 155345. 225122. 0. 0. 0.] 
[ 220930. 286375. 151266. 0. 0.] 
[ 151204. 192973. 302254. 151349. 394475.]]
Let's build the Embedding() layer in Keras, using pre-trained word vectors. After this layer is built, you will pass the output of sentences_to_indices() to it as an input, and the Embedding() layer will return the word embeddings for a sentence.

Exercise: Implement pretrained_embedding_layer(). You will need to carry out the following steps:

Initialize the embedding matrix as a numpy array of zeroes with the correct shape.
Fill in the embedding matrix with all the word embeddings extracted from word_to_vec_map.
Define Keras embedding layer. Use Embedding(). Be sure to make this layer non-trainable, by setting trainable = False when calling Embedding(). If you were to set trainable = True, then it will allow the optimization algorithm to modify the values of the word embeddings.
Set the embedding weights to be equal to the embedding matrix  (emoji.py)

Expected Output:

weights[0][1][3] =	-0.3403

2.3 Building the Emojifier-V2
Lets now build the Emojifier-V2 model. You will do so using the embedding layer you have built, and feed its output to an LSTM network. (refer image)

Figure 5: Emojifier-v2. A 2-layer LSTM sequence classifier.

Exercise: Implement Emojify_V2(), which builds a Keras graph of the architecture shown in Figure 3. The model takes as input an array of sentences of shape (m, max_len, ) defined by input_shape. It should output a softmax probability vector of shape (m, C = 5). You may need Input(shape = ..., dtype = '...'), LSTM(), Dropout(), Dense(), and Activation(). (refer emoji.py)

Run the following cell to create your model and check its summary. Because all sentences in the dataset are less than 10 words, we chose max_len = 10. You should see your architecture, it uses "20,223,927" parameters, of which 20,000,050 (the word embeddings) are non-trainable, and the remaining 223,877 are. Because our vocabulary size has 400,001 words (with valid indices from 0 to 400,000) there are 400,001*50 = 20,000,050 non-trainable parameters.  (refer emoji.py)

As usual, after creating your model in Keras, you need to compile it and define what loss, optimizer and metrics your are want to use. Compile your model using categorical_crossentropy loss, adam optimizer and ['accuracy'] metrics: (refer emoji.py)

It's time to train your model. Your Emojifier-V2 model takes as input an array of shape (m, max_len) and outputs probability vectors of shape (m, number of classes). We thus have to convert X_train (array of sentences as strings) to X_train_indices (array of sentences as list of word indices), and Y_train (labels as indices) to Y_train_oh (labels as one-hot vectors).  (refer emoji.py)

Fit the Keras model on X_train_indices and Y_train_oh. We will use epochs = 50 and batch_size = 32. (emoji.py)

Your model should perform close to 100% accuracy on the training set. The exact accuracy you get may be a little different. Run the following cell to evaluate your model on the test set. (refer emoji.py)

You should get a test accuracy between 80% and 95%. Run the cell below to see the mislabelled examples. (refer emoji.py)

Expected emoji:😄 prediction: she got me a nice present	❤️
Expected emoji:😄 prediction: Stop making this joke ha ha ha	😞
Expected emoji:🍴 prediction: any suggestions for dinner	😄
Expected emoji:😄 prediction: you brighten my day	😞
Expected emoji:⚾ prediction: enjoy your game❤️
Expected emoji:😄 prediction: she said yes	😞
Expected emoji:😄 prediction: will you be my valentine	😞
Expected emoji:😄 prediction: I like to laugh	❤️
Expected emoji:😄 prediction: What you did was awesome	😞
Expected emoji:😞 prediction: go away	⚾
Expected emoji:😞 prediction: yesterday we lost again	⚾
Expected emoji:❤️ prediction: family is all I have	🍴
Now you can try it on your own example. Write your own sentence below. (refer emoji.py)

not feeling happy 😞
Previously, Emojify-V1 model did not correctly label "not feeling happy," but our implementation of Emojiy-V2 got it right. (Keras' outputs are slightly random each time, so you may not have obtained the same result.) The current model still isn't very robust at understanding negation (like "not happy") because the training set is small and so doesn't have a lot of examples of negation. But if the training set were larger, the LSTM model would be much better than the Emojify-V1 model at understanding such complex sentences.

Congratulations!

What we should remember:

If you have an NLP task where the training set is small, using word embeddings can help your algorithm significantly. Word embeddings allow your model to work on words in the test set that may not even have appeared in your training set.
Training sequence models in Keras (and in most other deep learning frameworks) requires a few important details:
To use mini-batches, the sequences need to be padded so that all the examples in a mini-batch have the same length.
An Embedding() layer can be initialized with pretrained values. These values can be either fixed or trained further on your dataset. If however your labeled dataset is small, it's usually not worth trying to train a large pre-trained set of embeddings.
LSTM() has a flag called return_sequences to decide if you would like to return every hidden states or only the last one.
You can use Dropout() right after LSTM() to regularize your network.

















