import os
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import string
import re
import numpy as np
from multiprocessing import Pool
import simplemma
import fasttext

path_to_model = 'C:\\Users\\User\\Desktop\\python_x64\\lid.176.bin'
db_path = "D:\\DB\\DB"
prep_db_path = "D:\\DB\\prep"

_words_to_remove = stopwords.words('romanian')
_model = fasttext.load_model(path_to_model)


def any_in(arr, string):
	for i in arr:
		if i in string:
			return True
	return False


def if_entity(word):
	if word[0].isupper() == True:
		return  '__label__ro' in _model.predict(word, k=3)[0]


def prepocces_file(file, sent_tok=True):

	if not path.exists('prep_db_path'):
		os.makedir(prep_db_path)



	punctuation = "#$%&'()*+,-/:;<=>@[\]^_`{|}~" + '”' + '“' + '„' + '’' + "'" + '"' + ' ' + '‘' + '–' + '|' + '─' + '┤' + '├' + '─' + '┼' + '│' + \
									'┴' + '┘' + '└' + '┌' + '┐' + '┬' +'�' + '►' + '•' + '`' + '…' + '⌠' + '⌡' + '═' + '╝' + '╤' + '╚' + '←' + \
									'║' + '╔' + '╗' + '±' + '­-­' + '§' + '†' + '‡' + '' + '¦' + '—'


	langdata = simplemma.load_data('ro')


	with open(os.path.join(db_path, file), encoding='utf-8-sig') as f:
		text = f.read()
		text = re.sub('\d+,?\.?\d*', "", text)	
		text = re.sub(r' +', ' ', text) 

		if sent_tok == False:
			punctuation += '.?!'
			text = text.translate(str.maketrans(punctuation, ' ' * len(punctuation)))
			text = re.sub(r' +', ' ', text)
	
			tokens = [simplemma.lemmatize(word.lower(), langdata) for word in word_tokenize(text) if not word.lower() in _words_to_remove and not if_entity(word)]

			
			with open(os.path.join(prep_db_path, 'prep_' + file), 'w', encoding='utf-8-sig') as file:
				for token in tokens:
					file.write(token)
					file.write(' ')
		else:
			text = text.translate(str.maketrans(punctuation, ' ' * len(punctuation)))
			sentences = sent_tokenize(text)
			tokens = [[simplemma.lemmatize(word.lower(), langdata) for word in word_tokenize(sentence) if not any_in(['.', '?', '!'], word) and not (word.lower() in _words_to_remove)\
			and not if_entity(word)] for sentence in sentences]

			with open(os.path.join(prep_db_path, 'prep_' + file), 'w', encoding='utf-8-sig') as file:
				for sentence in tokens:
					for token in sentence:
						file.write(token+ ' ')
					file.write('\n')


def main():

	_words_to_remove.extend(['și'])
	files = [file for file in os.listdir(db_path) if file.endswith('txt')]

	with Pool(10) as p:
		p.map(prepocces_file, files)


if __name__ == '__main__':
	main()

