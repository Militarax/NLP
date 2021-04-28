import os
import re
import gensim
import random
import numpy as np
import rowordnet as rwn
import simplemma
import fasttext
import spacy
import simplemma
from scipy import spatial
from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer
from gensim.test.utils import datapath
from gensim.models.callbacks import CallbackAny2Vec
from word2vec_model import MonitorCallback


_path_to_fasttext_model = 'C:\\Users\\User\\Desktop\\python_x64\\lid.176.bin'

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



def save_train_data(vocab, model, neg_samples=1):
	langdata = simplemma.load_data('ro')

	with open('train_data.txt', 'a+', encoding='utf-8-sig') as train_data_file:
		with open('hypernyms.txt', 'r', encoding='utf-8-sig') as file:

			for line in file:
				words = line.split()
				word, hypernyms = words[0], words[1:]
				if len(hypernyms) != 0:
					for hypernym in hypernyms:
						if word != hypernym:
							train_data_file.write(word)
							train_data_file.write(',')

							train_data_file.write(hypernym)
							train_data_file.write(',')

							train_data_file.write(str(1) + '\n')

					for neg_sample in range(neg_samples * len(hypernyms)):
						random_word = vocab[random.randint(0, len(vocab) - 1)]

						if word != random_word:
							train_data_file.write(word)
							train_data_file.write(',')

							train_data_file.write(random_word)
							train_data_file.write(',')

							train_data_file.write(str(0) + '\n')



def noun(nlp, fasttext_model, word):
	doc = nlp(word)
	if len(doc) == 1:
		token = doc[0]
		if token.pos_ == "NOUN" and '__label__ro' in fasttext_model.predict(word, k=5)[0]:
			return True

	return False


def extract_hypernyms_rowordnet(vocab, nlp, fasttext_model, wn):
	langdata = simplemma.load_data('ro')

	for index, word in enumerate(vocab):
		if noun(nlp, fasttext_model, word):
			hypernyms = set()
			for synset_id in wn.synsets(literal=word, pos=rwn.Synset.Pos.NOUN):

				relations = wn.outbound_relations(synset_id)
				for relation in relations:
					target_synset_id = relation[0]
					rel = relation[1]
					if rel == 'hypernym':
						hypernyms = hypernyms | set(wn.synset(target_synset_id).literals)


			with open('hypernyms.txt', 'a+', encoding='utf-8-sig') as file:
				if len(hypernyms) != 0:
					file.write(word)
					for hypernym in hypernyms:
						if not ('_' in hypernym) and not('[' in hypernym) and simplemma.lemmatize(hypernym, langdata) in vocab:
							file.write(' ' + simplemma.lemmatize(hypernym, langdata))
					file.write('\n')
		print(index)

def extract_hyponyms_rowordnet(vocab, nlp, fasttext_model, wn):
	langdata = simplemma.load_data('ro')

	for index, word in enumerate(vocab):
		if noun(nlp, fasttext_model, word):
			hyponyms = set()
			for synset_id in wn.synsets(literal=word, pos=rwn.Synset.Pos.NOUN):

				relations = wn.relations(synset_id)
				for relation in relations:
					target_synset_id = relation[0]
					rel = relation[1]
					if rel == 'hyponym':
						hyponyms = hyponyms | set(wn.synset(target_synset_id).literals)


			with open('hyponyms.txt', 'a+', encoding='utf-8-sig') as file:
				if len(hyponyms) != 0:
					file.write(word)
					for hyponym in hyponyms:
						if not ('_' in hyponym) and not('[' in hyponym) and simplemma.lemmatize(hyponym, langdata) in vocab:
							file.write(' ' + simplemma.lemmatize(hyponym, langdata))
					file.write('\n')
		print(index)

def main():

	wn = rwn.RoWordNet()
	model = Word2Vec.load('D:\\DB\\model\\word2vec.model')
	nlp = spacy.load("ro_core_news_sm")
	fasttext_model = fasttext.load_model(_path_to_fasttext_model)
	
	try:
		vocab = get_vocab()
	except Exception as e:
		vocab = model.wv.vocab

	# extract_hypernyms_rowordnet(vocab, nlp, fasttext_model, wn)
	extract_hyponyms_rowordnet(vocab, nlp, fasttext_model, wn)
	# save_train_data(vocab, model)




if __name__ == '__main__':
	main()
