# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:32:49
# @File: abstract_dataset.py


import random
import os
import copy
import torch
from mwptoolkit.utils.utils import read_json_data, write_json_data, read_json_data_sig
from mwptoolkit.utils.preprocess_tools import get_group_nums, get_deprel_tree, get_span_level_deprel_tree
from mwptoolkit.utils.preprocess_tools import id_reedit
from mwptoolkit.utils.preprocess_tool.equation_operator import from_postfix_to_infix, from_prefix_to_infix, operator_mask,EN_rule1_stat,EN_rule2
from mwptoolkit.utils.enum_type import MaskSymbol
from mwptoolkit.utils.enum_type import DatasetName,FixType
from pororo import Pororo


class AbstractDataset(object):
    """abstract dataset

    the base class of dataset class
    """
    def __init__(self, config):
        """
        Args:
            config (mwptoolkit.config.configuration.Config)
        
        expected that config includes these parameters below:

        model (str): model name.

        dataset (str): dataset name.

        equation_fix (str): [infix | postfix | prefix], convert equation to specified format.
        
        dataset_path (str): the road path of dataset folder.

        language (str): a property of dataset, the language of dataset.

        single (bool): a property of dataset, the equation of dataset is single or not.

        linear (bool): a property of dataset, the equation of dataset is linear or not.

        source_equation_fix (str): [infix | postfix | prefix], a property of dataset, the source format of equation of dataset.

        rebuild (bool): when loading additional dataset infomation, this can decide to build infomation anew or load infomation built before.

        validset_divide (bool): whether to split validset. if True, the dataset is split to trainset-validset-testset. if False, the dataset is split to trainset-testset.

        mask_symbol (str): [NUM | number], the symbol to mask numbers in equation.
        
        min_word_keep (int): in dataset, words that count greater than the value, will be kept in input vocabulary.
        
        min_generate_keep (int): generate number that count greater than the value, will be kept in output symbols.

        symbol_for_tree (bool): build output symbols for tree or not.

        share_vocab (bool): encoder and decoder of the model share the same vocabulary, often seen in Seq2Seq models.

        k_fold (int|None): if it's an integer, it indicates to run k-fold cross validation. if it's None, it indicates to run trainset-validset-testset split.

        read_local_folds (bool): when running k-fold cross validation, if True, then loading split folds from dataset folder. if False, randomly split folds.

        shuffle (bool): whether to shuffle trainset before training.
        """
        super().__init__()
        self.model = config["model"]
        self.dataset = config["dataset"]
        self.equation_fix = config["equation_fix"]
        
        self.dataset_path = config["dataset_path"]
        self.language = config["language"]
        self.single = config["single"]
        self.linear = config["linear"]
        self.source_equation_fix = config["source_equation_fix"]

        self.rebuild = config['rebuild']
        self.validset_divide = config["validset_divide"]
        
        self.mask_symbol = config["mask_symbol"]
        self.min_word_keep = config["min_word_keep"]
        self.min_generate_keep = config["min_generate_keep"]
        self.symbol_for_tree = config["symbol_for_tree"]
        self.share_vocab = config["share_vocab"]
        
        self.k_fold = config["k_fold"]
        self.read_local_folds = config["read_local_folds"]
        self.shuffle = config["shuffle"]
        
        self.device = config["device"]
        
        self.root = config['root']
        
        self.pre_mask = config['pre_mask']
        self.mask_grouping = config['mask_grouping']

        self.prompt = config['prompt']
        self.mapping = config['mapping']
        self.encode_type = config['encode_type']

        self.mask_entity = config['mask_entity']

        self.max_span_size = 1

    def _load_dataset(self):
        '''
        read dataset from files
        '''
        if 'read_json_data' not in locals():
            try:
                from mwptoolkit.utils.utils import read_json_data, write_json_data, read_json_data_sig
            except:
                pass
        if self.encode_type == 'seg':
            read_json_data = read_json_data_sig

        trainset_file = self.dataset_path + "/trainset.json"
        validset_file = self.dataset_path + "/validset.json"
        testset_file = self.dataset_path + "/testset.json"
        
        if os.path.isabs(trainset_file):
            self.trainset = read_json_data(trainset_file)
        else:
            self.trainset = read_json_data(os.path.join(self.root,trainset_file))
        if os.path.isabs(validset_file):
            self.validset = read_json_data(validset_file)
        else:
            self.validset = read_json_data(os.path.join(self.root,validset_file))
        if os.path.isabs(testset_file):
            self.testset = read_json_data(testset_file)
        else:
            self.testset = read_json_data(os.path.join(self.root,testset_file))

        if self.validset_divide is not True:
            self.testset = self.validset + self.testset
            self.validset = []

        if self.dataset in ['kor_asdiv-a', 'kor_di_asdiv-a', DatasetName.hmwp]:
            self.trainset,self.validset,self.testset = id_reedit(self.trainset, self.validset, self.testset, id_key='ID')

        if self.mapping:
            mapping_file = self.dataset_path + '/mapping.txt'
            if os.path.isabs(mapping_file):
                mapping_table = read_mapping_table(mapping_file)
            else:
                mapping_table = read_mapping_table(os.path.join(self.root,mapping_file))
            
            for it in self.trainset:
                it['Question'] = mapping(it['Question'], mapping_table)
            for it in self.validset:
                it['Question'] = mapping(it['Question'], mapping_table)
            for it in self.testset:
                it['Question'] = mapping(it['Question'], mapping_table)
                        
        if self.prompt:
            prompt_file = self.dataset_path + "/prompt.txt"
            if os.path.isabs(prompt_file):
                prompt_table = read_prompt_table(prompt_file)
            else:
                prompt_table = read_prompt_table(os.path.join(self.root,prompt_file))
            regex = get_regex_from_from_table(prompt_table)
            
            for it in self.trainset:
                it['Question'] = prompting(it, prompt_table, regex)
            for it in self.validset:
                it['Question'] = prompting(it, prompt_table, regex)
            for it in self.testset:
                it['Question'] = prompting(it, prompt_table, regex)

        # if self.pre_mask == MaskSymbol.NUM or self.pre_mask == MaskSymbol.number:
        #     for it in self.trainset:
        #         question, num_list = num_masking(it['Question'], self.pre_mask, self.mask_grouping)
        #         equation = num_masking_const(it['Equation'], self.pre_mask, num_list)
        #
        #         it['Question'] = question
        #         it['Equation'] = equation
        #         it['Numbers'] = ' '.join(num_list)
        #     for it in self.validset:
        #         question, num_list = num_masking(it['Question'], self.pre_mask, self.mask_grouping)
        #         equation = num_masking_const(it['Equation'], self.pre_mask, num_list)
        #
        #         it['Question'] = question
        #         it['Equation'] = equation
        #         it['Numbers'] = ' '.join(num_list)
        #     for it in self.testset:
        #         question, num_list = num_masking(it['Question'], self.pre_mask, self.mask_grouping)
        #         equation = num_masking_const(it['Equation'], self.pre_mask, num_list)
        #
        #         it['Question'] = question
        #         it['Equation'] = equation
        #         it['Numbers'] = ' '.join(num_list)
        # elif self.pre_mask is not None:
        #     raise NotImplementedError

        self.trainset = cleanse_dataset(self.trainset)
        self.validset = cleanse_dataset(self.validset)
        self.testset = cleanse_dataset(self.testset)

    def _load_fold_dataset(self):
        """read one fold of dataset from file. 
        """
        if self.encode_type == 'seg':
            read_json_data = read_json_data_sig
        trainset_file = self.dataset_path + "/trainset_fold{}.json".format(self.fold_t)
        testset_file = self.dataset_path + "/testset_fold{}.json".format(self.fold_t)
        if os.path.isabs(trainset_file):
            self.trainset = read_json_data(trainset_file)
        else:
            self.trainset = read_json_data(os.path.join(self.root,trainset_file))
        if os.path.isabs(trainset_file):
            self.testset = read_json_data(testset_file)
        else:
            self.testset = read_json_data(os.path.join(self.root,testset_file))
        self.validset = []

    def fix_process(self, fix):
        r"""equation infix/postfix/prefix process.

        Args:
            fix (function): a function to make infix, postfix, prefix or None  
        """
        source_equation_fix=self.source_equation_fix if self.source_equation_fix else FixType.Infix
        if fix != None:
            for idx, data in enumerate(self.trainset):
                if source_equation_fix==FixType.Prefix:
                    self.trainset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.trainset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.trainset[idx]["infix equation"] = copy.deepcopy(data["equation"])
                self.trainset[idx]["equation"] = fix(data["equation"])
            for idx, data in enumerate(self.validset):
                if source_equation_fix==FixType.Prefix:
                    self.validset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.validset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.validset[idx]["infix equation"] = copy.deepcopy(data["equation"])
                self.validset[idx]["equation"] = fix(data["equation"])
            for idx, data in enumerate(self.testset):
                if source_equation_fix==FixType.Prefix:
                    self.testset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.testset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.testset[idx]["infix equation"] = copy.deepcopy(data["equation"])
                self.testset[idx]["equation"] = fix(data["equation"])
        else:
            for idx, data in enumerate(self.trainset):
                if source_equation_fix==FixType.Prefix:
                    self.trainset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.trainset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.trainset[idx]["infix equation"] = copy.deepcopy(data["equation"])
            for idx, data in enumerate(self.validset):
                if source_equation_fix==FixType.Prefix:
                    self.validset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.validset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.validset[idx]["infix equation"] = copy.deepcopy(data["equation"])
            for idx, data in enumerate(self.testset):
                if source_equation_fix==FixType.Prefix:
                    self.testset[idx]["infix equation"] = from_prefix_to_infix(data["equation"])
                elif source_equation_fix==FixType.Postfix:
                    self.testset[idx]["infix equation"] = from_postfix_to_infix(data["equation"])
                else:
                    self.testset[idx]["infix equation"] = copy.deepcopy(data["equation"])

    def operator_mask_process(self):
        """operator mask process of equation.
        """
        for idx, data in enumerate(self.trainset):
            self.trainset[idx]["template"] = operator_mask(data["equation"])
        for idx, data in enumerate(self.validset):
            self.validset[idx]["template"] = operator_mask(data["equation"])
        for idx, data in enumerate(self.testset):
            self.testset[idx]["template"] = operator_mask(data["equation"])

    def en_rule1_process(self, k):
        rule1_list = EN_rule1_stat(self.trainset, k)
        for idx, data in enumerate(self.trainset):
            flag = False
            equ_list = data["equation"]
            for equ_lists in rule1_list:
                if equ_list in equ_lists:
                    self.trainset[idx]["equation"] = equ_lists[0]
                    flag = True
                    break
                if flag:
                    break
        for idx, data in enumerate(self.validset):
            flag = False
            equ_list = data["equation"]
            for equ_lists in rule1_list:
                if equ_list in equ_lists:
                    self.validset[idx]["equation"] = equ_lists[0]
                    flag = True
                    break
                if flag:
                    break
        for idx, data in enumerate(self.testset):
            flag = False
            equ_list = data["equation"]
            for equ_lists in rule1_list:
                if equ_list in equ_lists:
                    self.testset[idx]["equation"] = equ_lists[0]
                    flag = True
                    break
                if flag:
                    break

    def en_rule2_process(self):
        for idx, data in enumerate(self.trainset):
            self.trainset[idx]["equation"] = EN_rule2(data["equation"])
        for idx, data in enumerate(self.validset):
            self.validset[idx]["equation"] = EN_rule2(data["equation"])
        for idx, data in enumerate(self.testset):
            self.testset[idx]["equation"] = EN_rule2(data["equation"])

    def cross_validation_load(self, k_fold, start_fold_t=0):
        r"""dataset load for cross validation

        Build folds for cross validation. Choose one of folds for testset and other folds for trainset.
        
        Args:
            k_fold (int): the number of folds, also the cross validation parameter k.
            start_fold_t (int): default 0, training start from the training of t-th time.
        
        Returns:
            Generator including current training index of cross validation.
        """
        if k_fold<=1:
            raise ValueError("the cross validation parameter k shouldn't be less than one, it should be greater than one")
        if self.read_local_folds != True:
            self._load_dataset()
            self.datas = self.trainset + self.validset + self.testset
            random.shuffle(self.datas)
            step_size = int(len(self.datas) / k_fold)
            folds = []
            for split_fold in range(k_fold - 1):
                fold_start = step_size * split_fold
                fold_end = step_size * (split_fold + 1)
                folds.append(self.datas[fold_start:fold_end])
            folds.append(self.datas[(step_size * (k_fold - 1)):])
        self.start_fold_t = start_fold_t
        for k in range(self.start_fold_t, k_fold):
            self.fold_t = k
            self.trainset = []
            self.validset = []
            self.testset = []
            if self.read_local_folds:
                self._load_fold_dataset()
            else:
                for fold_t in range(k_fold):
                    if fold_t == k:
                        self.testset += copy.deepcopy(folds[fold_t])
                    else:
                        self.trainset += copy.deepcopy(folds[fold_t])
            self._preprocess()
            self._build_vocab()
            if self.shuffle:
                random.shuffle(self.trainset)
            yield k

    def dataset_load(self):
        r"""dataset process and build vocab
        """
        self._load_dataset()
        self._preprocess()
        self._build_vocab()
        if self.shuffle:
            random.shuffle(self.trainset)

    def _preprocess(self):
        raise NotImplementedError

    def _build_vocab(self):
        raise NotImplementedError

    def _update_vocab(self, vocab_list):
        raise NotImplementedError

def read_prompt_table(prompt_file_path):
    with open(prompt_file_path, 'r') as f:
        table_string = f.read()
        
    return read_prompt_table_from_string(table_string)

from collections import defaultdict
def read_prompt_table_from_string(table_string):
    ret = defaultdict(list)
    
    for line in table_string.split('\n'):
        keywords, sentence = line.split('|')
        keyword_list = keywords.split(',')
        
        ret[tuple(sorted(keyword_list))].append(sentence)
    
    return [(set(key), value) for key, value in ret.items()]

import itertools
def get_regex_from_from_table(table):
    return re.compile('(' + '|'.join(sorted(set(itertools.chain(*[key for key, _ in table])), key=len)[::-1]) + ')')

import re
def get_prompt(question, table, regex):
    question_keyword_set = set(re.findall(regex, question))
    
    prompt = ' '.join([' '.join(sentences) for key, sentences in table if key.issubset(question_keyword_set)])
    
    if prompt:
        return prompt + ' '
    else:
        return ''

def read_mapping_table(mapping_file_path):
    with open(mapping_file_path, 'r') as f:
        table_string = f.read()
        
    return read_mapping_table_from_string(table_string)

from collections import defaultdict
def read_mapping_table_from_string(table_string):
    ret = defaultdict(list)
    
    all_keywords = []
    
    for line in table_string.split('\n'):
        keywords, mapped = line.split('|')
        keyword_list = keywords.split(',')
        
        for keyword in keyword_list:
            all_keywords.append((keyword, mapped))

    return all_keywords

def mapping(sentence, mapping_table):
    for keyword, mapped in mapping_table:
        sentence = sentence.replace(keyword, mapped)
    return sentence

def prompting(q, prompt_table, regex):
    prompt = get_prompt(q['Question'], prompt_table, regex)

    return prompt + q['Question']

number_regex = re.compile(r'\d+(?:\.\d+)?')
def num_masking(string, num_mask_type, mask_group):
    if num_mask_type == MaskSymbol.NUM and mask_group:
        raise Exception("NUM일경우 group 불가")

    numbers = []
    res = []
    for m in number_regex.finditer(string):
        start = m.start()
        end = m.end()

        if mask_group and m.group(0) in numbers:
            token_idx = numbers.index(m.group(0))
        else:
            token_idx = len(numbers)
            numbers.append(m.group(0))
        res.append((start, end, token_idx))


    for start, end, token_idx in res[::-1]:
        if num_mask_type == MaskSymbol.NUM:
            string = string[:start] + 'NUM' + string[end:]
        elif num_mask_type == MaskSymbol.number:
            string = string[:start] + f'NUM_{token_idx}' + string[end:]
        else:
            raise NotImplementedError

    return string, numbers

def num_masking_const(string, num_mask_type, numbers):
    res = []
    for m in number_regex.finditer(string):
        if m.group(0) in numbers:
            start = m.start()
            end = m.end()

            token_idx = numbers.index(m.group(0))
            res.append((start, end, token_idx))

    for start, end, token_idx in res[::-1]:
        if num_mask_type == MaskSymbol.NUM:
            string = string[:start] + 'NUM' + string[end:]
        elif num_mask_type == MaskSymbol.number:
            string = string[:start] + f'NUM_{token_idx}' + string[end:]
        else:
            raise NotImplementedError

    return string


def cleanse_dataset(dataset):
    new_dataset = []
    for data in dataset:
        data['Question'] = data['Question'].replace('\u200b', '')
        new_dataset.append(data)
    return new_dataset

