import pandas as pd
import numpy as np
import json
import os
import re
from spellchecker import SpellChecker
import ftfy

#change this to your name
sender = "Michael Lu"

output_file = "out.txt"

#change this path to be wherever your downloaded messages directory is
relative_path = 'messages_dir/inbox'

def iterate_over_working_directory(relative_path, output_file):
	out = open(output_file, "w")
	for subdir, dirs, files in os.walk(relative_path):
		for file in files:
			if (file.endswith('.json')):
				parse_json_and_write_to_txt(os.path.join(subdir, file), out)
	out.close()


def parse_json_and_write_to_txt(json_filename, out):
	with open(json_filename, 'r') as f:
		json_dict = json.load(f)

	partial_text_to_write = ""
	json_dict["messages"].reverse()
	for item in json_dict["messages"]:
		for x in item:
			if (x == 'content' and item['sender_name'] == sender):
				out.write(item[x] + '\n')
	f.close()

import re
output_file = "out.txt"
def read_text_file_and_create_map():
	vocab_to_count_map = {}
	with open(output_file, 'r') as f:
		for line in f:
			tokens = re.findall(r'\w+', line)
			for token in tokens:
				token = ftfy.fix_text(token)
				if (token in vocab_to_count_map):
					vocab_to_count_map[token] = vocab_to_count_map[token] + 1
				else: 
					vocab_to_count_map[token] = 1
	f.close()
	return vocab_to_count_map

# vocab: map<word, frequency count>
def spellcheck_vocab(vocab):
	# correct --> likely misspelled spellings (WANT)
	typo_dict = {}
	spell = SpellChecker()

	# Find the words that may be misspelled
	misspelled = spell.unknown(set(vocab.keys()))
	print(len(misspelled))
	for typo in misspelled:
		# Get a list of `likely` options
		word = spell.correction(typo)
		if (word in vocab):	
			if word not in typo_dict:
				typo_dict[word] = set()
			typo_dict[word].add(typo)
	output = {}

	
	for word in typo_dict:
		misspellings = typo_dict[word]
		total_occur = float(vocab[word])
		word_map = {}
		for typo in misspellings:
			if (typo in vocab and typo is not word):
				total_occur += vocab[typo]
				word_map[typo] = vocab[typo]
		word_map[word] = float(vocab[word])

		# Divide everything in map by total_occur
		word_map = {k:v/total_occur for (k,v) in word_map.items()}
		# Store in typo_dict
		output[word] = word_map

	return output

iterate_over_working_directory(relative_path, output_file)
vocab_to_count_map = read_text_file_and_create_map()
output = spellcheck_vocab(vocab_to_count_map)
print(output)
