import pandas as pd
import numpy as np
import json
import os
#using this to parse the unicode text into emojis
import ftfy
import re
import codecs

#change this to your name
sender = "Michael Lu"

output_file = "out.txt"

#change this path to be wherever your downloaded messages directory is
relative_path = 'messages_dir/inbox'

RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

def iterate_over_working_directory(relative_path, output_file):
  out = open(output_file, "w")
  for subdir, dirs, files in os.walk(relative_path):
    for file in files:
      if (file.endswith('.json')):
        #parse_json_and_write_to_txt(os.path.join(subdir, file), out)
        #print(os.path.join(subdir, file), out)
        #parse_single_user_emojis(os.path.join(subdir, file), out)
        parse_json_and_write_other_members_names_txt(os.path.join(subdir, file), out)
  out.close() 


def parse_json_and_write_to_txt(json_filename, out):
  with open(json_filename, 'r') as f:
    json_dict = json.load(f)

  partial_text_to_write = ""
  json_dict["messages"].reverse()
  for item in json_dict["messages"]:
    for x in item:
      if (x == 'content'):
        out.write(ftfy.fix_text(item["sender_name"] + ": " + item[x] + '\n'))
  f.close()
  return

def parse_json_and_write_other_members_names_txt(json_filename, out):
  with open(json_filename, 'r') as f:
    json_dict = json.load(f)
  
  names = ""
  for i in range(len(json_dict["participants"])):
    if (json_dict["participants"][i]["name"] == sender):
        continue
    if (i == len(json_dict["participants"]) - 1): 
        names = names + json_dict["participants"][i]["name"]
    else:
        names = names + json_dict["participants"][i]["name"] + ","
  names.strip(',')
  json_dict["messages"].reverse()
  for item in json_dict["messages"]:
    for x in item:
      if (x == 'content'):
        out.write(ftfy.fix_text(names + ": " + item[x] + '\n'))
  f.close()
  return

  #for i in range(len(json_dict["participants"])):
   # print(json_dict["participants"][i]["name"])



def strip_emoji(text):
    return RE_EMOJI.findall(text)

def read_txt(text_file):
    out_file = open("emoji_file.txt", "w")
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            list_of_emojis = strip_emoji(line)
            if (len(list_of_emojis) > 0):
                out_file.write(line)
    f.close()
    out_file.close()

iterate_over_working_directory(relative_path, output_file)
read_txt('out.txt')

#print(s)
#text = RE_EMOJI.findall(text)
#list_of_emojis = strip_emoji(text)
#print(list_of_emojis)