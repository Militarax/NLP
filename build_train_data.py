import os
import gensim
import numpy as np
from gensim.test.utils import datapath
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import re
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import simplemma
import fasttext
import spacy
from spacy.lang.ro.examples import sentences 
from scipy import spatial


_path_to_fasttext_model = 'C:\\Users\\User\\Desktop\\python_x64\\lid.176.bin'


class MonitorCallback(CallbackAny2Vec):
	def __init__(self):
		self.epoch = 0
	def on_epoch_begin(self, model):
		print("Epoch #{} start".format(self.epoch))

	def on_epoch_end(self, model):
		print("Epoch #{} end".format(self.epoch))
		self.epoch += 1


def pipeline(word):
	langdata = simplemma.load_data('ro')
	
	word = re.sub('\d+,?\.?\d*', "", word.lower())
	word = re.sub(r' +', ' ', word)
	
	return simplemma.lemmatize(word, langdata)


def build_dl_model(embedding_matrix):
	dl_model = keras.Sequential([
			keras.layers.Embedding(vocab_size, output_dim=100, weights=[embedding_matrix], input_length=2),
			keras.layers.Flatten(),
			keras.layers.Dense(1, activation='sigmoid')
		])

	dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	return dl_model


def get_embedding_matrix(tokenizer, vocab, model):

	embedding_matrix = np.zeros((vocab_size, 100))
	for word, i in tokenizer.word_index.items():
		if i in vocab:
			embedding_matrix[i] = model.wv[i]

	return embedding_matrix


def save_vocab_file():
	with open('D:\\DB\\vocab.txt', 'w', encoding='utf-8-sig') as file:
		for i in model.wv.vocab:
			file.write(i)
			file.write('\n')


def get_vocab():
	vocab = list()
	with open('D:\\DB\\vocab.txt', 'r', encoding='utf-8-sig') as file:
		vocab = file.read().split('\n')

	return vocab


def save_train_data(vocab, model):
	with open('D:\\DB\\train_data.txt', 'w', encoding='utf-8-sig') as file:
		for i in vocab:
			file.write(i)
			for similar_word in model.wv.most_similar(i, topn=5):
				file.write(' ' + similar_word[0])
			file.write('\n')


def filter_nouns(vocab, model):
	nlp = spacy.load("ro_core_news_sm")
	_model = fasttext.load_model(path_to_fasttext_model)
	romanian_words = []

	with open('D:\\DB\\train_data.txt', 'r', encoding='utf-8-sig') as file:
		for index, line in enumerate(file):
			word = line.split()[0]
			doc = nlp(word)
			for token in doc:
				if token.pos_ == "NOUN" and '__label__ro' in _model.predict(word, k=5)[0]:
					romanian_words.append(line)
				
	
	with open('D:\\DB\\train_data_nouns.txt','w', encoding='utf-8-sig') as file:
		for word in romanian_words:
			file.write(word)


def cos_similarity(model, word1, word2):
	return 1 - spatial.distance.cosine(model.wv[word1], model.wv[word2])

def main():


	model = Word2Vec.load('D:\\DB\\model\\word2vec.model')
	vocab_size = len(model.wv.vocab)
	
	
	try:
		vocab = get_vocab()
	except Exception as e:
		vocab = model.wv.vocab

	numbers_of_hypernyms = 5
	intersection_rate = 1
	number_of_sim_words = 100


	cleaned_word = pipeline('alb')
	print(cleaned_word)

	if cleaned_word in vocab:




		sims = model.wv.most_similar(cleaned_word, topn=number_of_sim_words)
		common_words = dict()
		
		while numbers_of_hypernyms != 0:
			for similar_word, similarity in sims:
				for co_hyp, co_sim in model.wv.most_similar(similar_word, topn=numbers_of_hypernyms):
					if co_hyp != cleaned_word:
						if co_hyp in common_words.keys():
							common_words[co_hyp] += 1
						else:
							common_words[co_hyp] = 1


			max_freq = max(list(map(lambda elem : elem[1], common_words.items())))

			saved_words = []
			for k, v in common_words.items():
				if v >= intersection_rate:
					saved_words.append([k,  cos_similarity(model, k, cleaned_word)])
			print(sorted(saved_words, key=lambda x : x[1]))
			break




	else:
		print('Not in vocabulary')



	# tokenizer = Tokenizer()
	# tokenizer.fit_on_texts('D:\\DB\\prep\\corpus.txt')

	# embedding_matrix = get_embedding_matrix(tokenizer, vocab, model)
	# dl_model = build_dl_model(embedding_matrix)




if __name__ == '__main__':
	main()
