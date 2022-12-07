# python interface

This repository is prepared for the python interface of this webserver.

Now, it’s just a primary version and still need to develop related functions and interfaces.

## 1. how to run?

### 1.1 pretrain model download

Firstly, you should download pretrain model from relevant github repository.

For example, if you want to use [DNAbert](https://github.com/jerryji1993/DNABERT), you need to put them into the pretrain folder and rename the relevant choice in the model.

### 1.2 start train

The entrance to the training model is in main folder, open it and you can easy know how to use it.

## 2. Models

We have collected more than ten common models applied to sentence classification, including but not limited to traditional methods. They are all in model folder and you can review it.

All the model input is sequences and output is divided into to two parts including classification output and feature representation. You can add your model in this format.

Here are our statistical table and related references.

|            NLP            |  traditional  |   GNN   |
| :-----------------------: | :-----------: | :-----: |
|    TransformerEncoder     |      DNN      | TextGNN |
|      ReformerEncoder      |      RNN      |   GCN   |
|     PerformerEncoder      |     LSTM      |   GAN   |
|     LinformerEncoder      |    BiLSTM     |         |
| RoutingTransformerEncoder | LSTMAttention |         |
|         DNA bert          |      GRU      |         |
|         Prot bert         |    TextCNN    |         |
|                           |   TextRCNN    |         |
|                           |     VDCNN     |         |
|                           |    RNN_CNN    |         |

## 3. Plot

we use matplotlib as our main plot tool.
