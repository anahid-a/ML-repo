import numpy as np
import re
from sklearn import svm
from matplotlib import pyplot as plt
import scipy.io as sp
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
def process_email(email_txt, vocab_list):
    email_txt = str(email_txt)
    email_txt = email_txt.lower()
    string = re.sub('[+><]', ' ', email_txt)
    string = re.sub('[0-9]+','number', string)
    string = re.sub('(http|https)://[^\s]*', 'httpaddr', string)
    string = re.sub('[^\s]+@[^\s]+', 'emailaddr', string)
    string = re.sub('[$]+', 'dollar', string)
    string = re.sub('[^a-zA-Z0-9]',' ',string)
    ps = PorterStemmer()
    words = word_tokenize(string)
    stems = []
    for w in words:
        stems.append(ps.stem(w))
    n_vocab = len(vocab_list)
    word_index = []
    for w in stems:
        for i in range(n_vocab):
            if w == vocab_list[i][1]:
                word_index.append(vocab_list[i][0])

    return word_index, stems

def email_features(word_index, vocab_list):
    n_vocab = len(vocab_list)
    features = np.zeros((n_vocab,1))
    for indx in word_index:
        features[indx-1, 0] = 1
    return features



def main():
    file_obj = open('emailSample1.txt','r')
    email_txt = file_obj.readlines()
    file_obj.close()
    datafrm = pd.read_csv('vocab.txt',sep='\t',header= None)
    vocab_list = datafrm.values.tolist()
    train_data = sp.loadmat('spamTrain.mat')
    x_mat = train_data["X"]
    y_mat = train_data["y"]
    test_data = sp.loadmat('spamTest.mat')
    x_test = test_data["Xtest"]
    y_test = test_data["ytest"]
    index, stems = process_email(email_txt, vocab_list)
    features = email_features(index, vocab_list)
    model = svm.SVC(C = 0.1, kernel='linear')
    n_samples = x_mat.shape[0]
    print("Training..............................")
    model.fit(x_mat, np.reshape(y_mat, (n_samples,)))
    prediction = model.predict(x_mat)
    print("Training set accuracy:")
    accuracy = np.mean(1*(prediction.T==y_mat.reshape(prediction.shape)))
    print(accuracy)
    print("Testing set accuracy:")
    prediction = model.predict(x_test)
    accuracy = np.mean(1*(prediction.T==y_test.reshape(prediction.shape)))
    print(accuracy)
    # Classify an email using the trained model
    file_obj = open('spamSample1.txt','r')
    email_txt = file_obj.readlines()
    file_obj.close()
    index, stems = process_email(email_txt, vocab_list)
    features = email_features(index, vocab_list)
    print("Sample email classification:")
    prediction = model.predict(features.reshape((1,len(vocab_list))))
    print(prediction)
if __name__=='__main__':
    main()