import os
from dataclasses import dataclass
import pickle
import numpy as np

@dataclass
class DataEngineering(object):
  '''
   prepare an input dataset for character-level language modeling
  '''
  datapath = '/home/ubuntu/nanoGPT/data/data/input.txt' #datapath

  @property
  def read_data(self):
      '''
       read the data
      '''
      with open(self.datapath, 'r') as f:
          data = f.read()

      return data

  @property
  def meta(self):
      '''
        save meta data to a pickle fine
      '''
      data = self.read_data
      print(f"length of dataset in characters: {len(data):,}")
      unq_chars = sorted(list(set(data)))
      vocab_size = len(unq_chars)
      print("all the unique characters:", ''.join(unq_chars))
      print(f"vocab size: {vocab_size:,}")
      stoi = { ch:i for i,ch in enumerate(unq_chars) }
      itos = { i:ch for i,ch in enumerate(unq_chars) }

      meta_data = {
             'vocab_size': vocab_size,
             'stoi': stoi,
             'itos': itos
              }
      with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta_data, f)

      return stoi, itos


  def encode(self, s: str):
      '''
        input a string and output a list of integers
        arg:
          s: str
        output:
          a list of integers
      '''
      stoi, itos = self.meta

      return [stoi[c] for c in s]

  def decoder(self, lst: list):
      '''
       input a list of integers and output a string
       arg:
         lst: list
       output:
         a string
      '''
      stoi, itos = self.meta
      
      return ''.join([itos[i] for i in l])
  
  def train_test_split(self, ratio=0.9):
      '''
        split a dataset into train/test with a certain ratio
        arg:
         ratio: split ratio
        output:
          training data and validation data
      '''
      data = self.read_data
      train_data = data[:int(len(data)*ratio)]
      val_data = data[int(len(data)*ratio):]

      return train_data, val_data

  @property
  def save_encoding(self):
      '''
       encode train and test to integers and save them to bin files and also pickle the meta data
      '''
      train_data, val_data = self.train_test_split()
      train_ids = self.encode(train_data)
      val_ids = self.encode(val_data)
      print(f"train has {len(train_ids):,} tokens")
      print(f"val has {len(val_ids):,} tokens")

      # save to bin files
      train_ids = np.array(train_ids, dtype=np.uint16)
      val_ids = np.array(val_ids, dtype=np.uint16)
      train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
      val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))



if __name__=='__main__':
    obj = DataEngineering()
    print(obj.meta)
    obj.save_encoding

