## A simple seq2seq model



#### Introduction

​	Environment : python3.5 + keras 2.0.5+ Tensorflow 1.0.1

​	I try building a seq2seq model with the environment above. Then I apply it on a simple word-transforming task. The method of transforming is just like the rule of transforming a noun to its plural form. To be more specific, the following rules:

> 1. Add a "s" to the end of the word.
> 2. If the word is end with "s", "x", "ch", "sh" or "o",  add "es" to the end of the word.
> 3. If the word is end with a consonant then a "y",  remove the "y" and add "ies" to the end of the word.
> 4. If the word is end with "f" or "fe",  remove the "f" or  "fe" and add "ves" to the end of the word.
> 5. If the word is end with "man", transform the "man" to "men"

​	The priority of the five rules increase and for one word only one rule can be used.

​	The training dataset and testing dataset are common English words. I get 5000 common words and randomly pick 80% of them for training, the other for testing.

​	To run the model, you should run `dataprepare.py` first to prepare the training dataset and testing dataset. Then run `seq2seq.py` with command-line parameter like this:

```
>>> python3.5 seq2seq.py train 300
>>> python3.5 seq2seq.py test
```

​	The "300" after parameter "train" is the number of epoch. And if you don't need to apply the model to the whole testing dataset, you can run this command:

```
>>> python3.5 seq2seq.py test "xxx"
```

​	So that you will get the predict result of the word "xxx".



#### Model

```
    # Initialize a sequential model
    model = Sequential()  
    
    # Encoder(a LSTM), pass the status of hidden layer to next time step. We need the
    # output from the last time step.
    model.add(LSTM(input_dim=input_size, output_dim=hidden_size, return_sequences=False))
    
    model.add(Dense(hidden_size, activation="relu"))
    
    # To use the LSTM model provided by Keras, I do some simplification here: using the
    # output of Encoder as the input of Decoder at each time step. So I need a 
    # RepeatVector to duplicate the output of Encoder
    model.add(RepeatVector(max_out_seq_len))
    
    # Decoder(another LSTM), output the final result
    model.add(LSTM(hidden_size, return_sequences=True))
    
    model.add(TimeDistributed(Dense(output_dim=input_size, activation="linear")))
    model.compile(loss="mse", optimizer='adam')
```



#### Result

​	I run the model with a hidden layer having 1024 nodes. And the number of epoch is 300.  The final accuracy is `0.830`. 

​	The model gives a good prediction on the short words. However, when handling long words, the performance will decrease. For example, there are some wrong result:

```
theological => theoooticals 
projection => procections
calculation => callulations 
reliability => relibiilities 
```

​	Most of them make a wrong letter in the word, and nearly all of the wrong results still end with "s" or "es". I think that's because I didn't take the Decoder's output at each time step as the input of the next time step, that decrease the influence of the output's context to Decoder. So there are still something to do for improvement.



#### Analysis and improvement

​	I did some simplification for using the LSTM model provided by Keras as mentioned above. That method just made a slight effect on a simple task like word-transforming. However, in the next work of translating or chatting based on a bigger corpus, it is necessary to modify the model. I think the implementation made by farizrahman4u is worth learning.

​	When doing vectorization of a word, I just use one-hot method for a higher accuracy rate. When doing some more complicate tasks like translating or chatting, the embedding method is also necessary.



#### Reference

​	[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)

​	[Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)

​	[farizrahman4u's seq2seq model](https://github.com/farizrahman4u/seq2seq)

