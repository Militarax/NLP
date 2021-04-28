import simplemma
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from word2vec_model import MonitorCallback, _EMBEDDING_DIM
from gensim.models import Word2Vec
from build_train_data import get_vocab
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MyTokenizer(object):

	def __init__(self, num_words=5000, separator=' '):
		self.num_words = num_words
		self.word_index = dict()
		self.separator = separator

	def tokenize_vocab(self, vocab):
		word_number = 1
		for word in vocab:
			self.word_index[word] = word_number
			word_number += 1 


	def fit_on_text_file(self, path_to_file, enc):
		word_number = 1
		with open(path_to_file, 'r', encoding=enc) as file:
			for line in file:
				for word in line.split():
					if not (word in self.word_index.keys()):
						self.word_index[word] = word_number
						word_number += 1


	def text_to_seq(self, text):
		seq = []
		for word in text.split(self.separator):
			if word in self.word_index.keys():
				seq.append(self.word_index[word])
			else:
				seq.append(0)

		return seq



def build_dl_model(embedding_matrix):
	dl_model = keras.Sequential([
			keras.layers.Embedding(embedding_matrix.shape[0], output_dim=_EMBEDDING_DIM, weights=[embedding_matrix], input_length=2),
			keras.layers.Flatten(),
			keras.layers.Dense(1, activation='sigmoid')
		])

	return dl_model


def get_embedding_matrix(tokenizer, vocab, model):
	vocab_size = len(vocab) + 1

	embedding_matrix = np.zeros((vocab_size, _EMBEDDING_DIM))
	for word, i in tokenizer.word_index.items():
		if word != '':
			embedding_matrix[i] = model.wv[word]

	return embedding_matrix


def get_train_data():
	words_1 = []
	words_2 = []
	Y_s = []
	with open('train_data.txt', 'r', encoding='utf-8-sig') as file:
		for line in file.read().split('\n'):
			line = line.split(',')

			if len(line) != 1:
				word_1 = line[0]
				word_2 = line[1]
				Y = int(line[2])
				words_1.append(word_1)
				words_2.append(word_2)
				Y_s.append(Y)

	return pd.DataFrame({'w1' : words_1, 'w2' : words_2, 'Y' : Y_s})




def main():
	model = Word2Vec.load('D:\\DB\\model\\word2vec.model')
	
	try:
		vocab = get_vocab()
	except Exception as e:
		vocab = model.wv.vocab
	
	langdata = simplemma.load_data('ro')
	
	tokenizer = MyTokenizer()
	tokenizer.tokenize_vocab(vocab)


	embedding_matrix = get_embedding_matrix(tokenizer, vocab, model)
	dl_model = build_dl_model(embedding_matrix)

	dl_model.summary()
	
	data = get_train_data()
	data = shuffle(data)
	
	train, test = train_test_split(data, test_size=0.2)

	X_train = np.array([[tokenizer.text_to_seq(train['w1'][index]), tokenizer.text_to_seq(train['w2'][index])] for index in train.index.values])
	Y_train = np.array(train['Y'])

	X_test = np.array([[tokenizer.text_to_seq(test['w1'][index]), tokenizer.text_to_seq(test['w2'][index])] for index in test.index.values])
	Y_test = np.array(test['Y'])

	callback = keras.callbacks.EarlyStopping(patience=5, verbose=1)
	dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	dl_model.fit(X_train, Y_train, batch_size=64, epochs=3, validation_split=0.2, verbose=1, callbacks=[callback], use_multiprocessing=True)
	
	predictions = dl_model.predict(X_test)

	print(accuracy_score(Y_test, predictions.round()))



if __name__ == '__main__':
	main()
