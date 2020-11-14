
import numpy as np
from os import path, makedirs
import zipfile

ZIP_FOLDER = './data'
if not path.exists(ZIP_FOLDER):
    makedirs(ZIP_FOLDER, exist_ok=True)


with zipfile.ZipFile('./sentiment_analysis_dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('./data/')

with open('./data/sentiment_analysis_dataset/reviews.txt' , 'r') as f:
    reviews = f.read()
    
with open('./data/sentiment_analysis_dataset/labels.txt', 'r') as f:
    labels = f.read()


print(reviews[:1000])
print(labels[:20])



# Data preprocessing
from string import punctuation

reviews = reviews.lower()
all_text = ''.join([c for c in reviews if c not in punctuation])



# split new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)


words = all_text.split()


# encoding the words
from collections import Counter

counts = Counter(words)
vocab = sorted(counts,key=counts.get,reverse=True)
vocab_to_int = {word:ii for ii,word in enumerate(vocab,1)}

reviews_int = []
for review in reviews_split:
    reviews_int.append([vocab_to_int[word] for word in review.split()])


# encoding the labels
labels_split = labels.split('\n')
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])


# outlier review stats
review_lens = Counter([len(x) for x in reviews_int])


# get indices of any reviews with length 0

non_zero_idx = [ii for ii,review in enumerate(reviews_int) if len(review) != 0]

# remove 0-length reviews and their labels
reviews_int = [reviews_int[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])


# padding sequences
def pad_features(reviews_int , seq_length):
    features = np.zeros((len(reviews_int) , seq_length) , dtype=int)
    
    for i,row in enumerate(reviews_int):
        features[i , -len(row):] = np.array(row)[:seq_length]
        
    return features


seq_length = 200

features = pad_features(reviews_int , seq_length)

assert len(features) == len(reviews_int)
assert len(features[0]) == seq_length





split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)

split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

## print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))





import torch
from torch.utils.data import TensorDataset ,DataLoader

# create a Tensor dataset
train_data = TensorDataset(torch.from_numpy(train_x) , torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x) , torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x) , torch.from_numpy(test_y))

# batchsize
batch_size = 50

train_loader = DataLoader(train_data , batch_size=batch_size)
valid_loader = DataLoader(valid_data , batch_size=batch_size)
test_loader = DataLoader(test_data , batch_size=batch_size)





# obtain one batch of training
dataiter = iter(train_loader)
sample_x , sample_y = dataiter.next()





print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)





# First checking if GPU is available
train_on_gpu = torch.cuda.is_available()

if (train_on_gpu):
    print("Training on GPU")
    
else:
    print("No GPU is available .. training on CPU")




import torch.nn as nn

class SentimentRNN(nn.Module):
    
    def __init__(self,vocab_size , output_size , embedding_dim , hidden_dim , n_layers,drop_prb = 0.5):
        
        super(SentimentRNN ,self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim , hidden_dim,n_layers,dropout=drop_prb,batch_first=True)
        
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(hidden_dim , output_size)
        self.sig = nn.Sigmoid()
        
    def forward(self,x,hidden):
        
        batch_size = x.size(0)
        
        embeds = self.embedding(x)
        lstm_out , hidden = self.lstm(embeds,hidden)
        
        lstm_out = lstm_out.contiguous().view(-1 , self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        sig_out = self.sig(out)
        
        sig_out = sig_out.view(batch_size,-1)
        sig_out = sig_out[:,-1]
        
        return sig_out , hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden





# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int) + 1
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size,output_size,embedding_dim,hidden_dim,n_layers)

print(net)





# Training
lr = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters() , lr = lr)





# training params

epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))





# Testing the model
# Get test data loss and accuracy

test_losses = []
num_correct = 0

h = net.init_hidden(batch_size)

net.eval()

for inputs,labels in test_loader:
    h = tuple([each.data for each in h])
    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
        
    output,h = net(inputs,h)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    pred = torch.round(output.squeeze())
    
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)
    
# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))





# negative test review
test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'





from string import punctuation

def tokenize_review(test_review):
    test_review = test_review.lower()
    
    test_text = ''.join([c for c in test_review if c not in punctuation])
    
    test_words = test_text.split()
    
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])
    
    return test_ints

# test code and generate tokenized review
test_ints = tokenize_review(test_review_neg)
print(test_ints)





# test sequence padding
seq_length = 200

features = pad_features(test_ints,seq_length)

print(features)




# test conversion to tensor and pass into your model

feature_tensor = torch.from_numpy(features)
print(feature_tensor.size())





def predict(net , test_review , seq_length = 200):
    
    net.eval()
    
    test_ints = tokenize_review(test_review)
    seq_length=seq_length
    
    features = pad_features(test_ints , seq_length)
    
    feature_tensor = torch.from_numpy(features)
    batch_size = feature_tensor.size(0)
    
    # initialize hidden state
    h = net.init_hidden(batch_size)
    
    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()
    # get the output from the model
    output, h = net(feature_tensor, h)
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze()) 
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    
    # print custom response
    if(pred.item()==1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")
        





# positive test review
test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'





# call function
seq_length=200 # good to use the length that was trained on

predict(net, test_review_neg, seq_length)







