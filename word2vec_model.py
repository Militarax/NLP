import os
import gensim 
from gensim.test.utils import datapath
from gensim.models.callbacks import CallbackAny2Vec

prep_db_path = "D:\\DB\\prep"
_EMBEDDING_DIM = 100

class MonitorCallback(CallbackAny2Vec):
	def __init__(self):
		self.epoch = 0
	def on_epoch_begin(self, model):
		print("Epoch #{} start".format(self.epoch))

	def on_epoch_end(self, model):
		print("Epoch #{} end".format(self.epoch))
		self.epoch += 1


def main():
	files = [file for file in os.listdir(prep_db_path) if file.endswith('txt')]
	
	with open(os.path.join(prep_db_path, 'corpus.txt'), 'a+', encoding='utf-8-sig') as f1:
		for index, file in enumerate(files):
			with open(os.path.join(prep_db_path, file), encoding='utf-8-sig') as f2:
				text = f2.read()
				f1.write(text)

	monitor = MonitorCallback()
	sentences = gensim.models.word2vec.LineSentence(datapath(os.path.join(prep_db_path, 'corpus.txt')))
	model = gensim.models.Word2Vec(sentences, size=_EMBEDDING_DIM, window=7, min_count=20, sg=1, negative=10, workers=10, iter=3, callbacks=[monitor])
	
	model.save("D:\\DB\\model\\word2vec.model")


if __name__ == '__main__':
	main()
