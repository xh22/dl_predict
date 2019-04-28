import numpy as np
import random
import re
import pickle
from rdkit import Chem
import sys
import time
import torch
from torch.utils.data import Dataset

class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    def __init__(self, init_from_file=None, max_length=140):
        self.special_tokens = ['EOS', 'GO']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        self.aa=[]
        if init_from_file: self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = [] 
        for i, char in enumerate(char_list):
            try:
               smiles_matrix.append(self.vocab[char])
            except:
                self.aa.append(char)
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['EOS']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        print(smiles)
        try:
            smiles = replace_halogen(smiles)
        except:
            print("FFFFF")
            print(smiles)
            print(type(smiles))
            exit()
        char_list = re.split(regex, smiles)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append('EOS')
        #tokenized += ['EOS' for i in range(140 - len(tokenized))]
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)



def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string

