
import pytorch_lightning as pl
import subprocess
import os
import json as js
import re

from transformers import AutoTokenizer, AutoModel

class BERTSimEncoderDataModule(pl.LightningDataModule):

	_sesame_repo_url = 'https://github.com/FAU-Inf2/sesame.git'

	def __init__(self, download = True, samples = 100):
		super().__init__()

		self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
		self.model = AutoModel.from_pretrained('bert-base-cased', output_hidden_states = True)
		self.download = download
		self.samples = samples

	def prepare_data(self):
		"""
		Data preparation is done in the next
		"""
		if self.download:
			# TODO: Add download

			return

		process = subprocess.run(['git', 'clone', self._sesame_repo_url], check = True)
		process = subprocess.run(['cd', 'sesame/src', '&&', 'make', 'pull'], check = True)

		file = open('sesame/dataset.json', 'r')

		data = js.load(file)

		file.close()

		os.mkdir('methods')

		func_pattern = re.compile("[a-zA-Z]*\\s*[a-zA-Z]+\\s+([_a-zA-Z]+)\\s*?\\(.*?\\)\\s*?\\{")

		repos_path = 'sesame/src/'

		for record in data:
			first_file_path = repos_path + record['first']['file']
			second_file_path = repos_path + record['second']['file']

			first_method_name = record['first']['method']






if __name__ == '__main__':
	print('Hello World')
