# -*- encoding: utf-8 -*-
# @Author: Junwon Hwang
# @Time: --
# @File: korean_dataset.py


import os
import copy
import warnings
from logging import getLogger
import re
import itertools
from collections import OrderedDict

import torch
from transformers import RobertaTokenizer,BertTokenizer,AutoTokenizer
from pororo import Pororo
import stanza
import pickle
import random


from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.data.dataset.pretrain_dataset import PretrainDataset
from mwptoolkit.utils.enum_type import DatasetName, MaskSymbol, NumMask,TaskType,FixType,Operators,SpecialTokens
from mwptoolkit.utils.preprocess_tools import id_reedit
from mwptoolkit.utils.preprocess_tool.equation_operator import from_infix_to_multi_way_tree
from mwptoolkit.utils.preprocess_tool.equation_operator import from_infix_to_postfix, from_infix_to_prefix, from_postfix_to_infix, from_postfix_to_prefix, from_prefix_to_infix, from_prefix_to_postfix
from mwptoolkit.utils.preprocess_tool.sentence_operator import deprel_tree_to_file, get_group_nums_, span_level_deprel_tree_to_file, get_span_level_deprel_tree_, get_deprel_tree_
from mwptoolkit.utils.preprocess_tool.number_transfer import seg_and_tag_svamp, get_num_pos
from mwptoolkit.utils.utils import write_json_data, read_json_data, str2float, lists2dict


class KoreanRobertaDataset(PretrainDataset):
    def shake_token(self, data):
        for it in data:
            Question = it['Question']
            ques_source_1 = it['ques source 1']
            Equation = it['Equation']
            entity_list = it['entity list']
            question = it['question']
            equation = it['equation']
            number_list = it['number list']
            infix_equation = it['infix equation']
            template = it['template']

            order = list(range(len(number_list)))
            random.shuffle(order)
            entity_order = list(range(len(entity_list)))
            random.shuffle(entity_order)

            number_tokens = {f'NUM_{from_}': f'NUM_{to}' for from_, to in enumerate(order)}
            number_order = {from_: to for from_, to in enumerate(order)}
            entity_tokens = {f'ETY_{from_}': f'ETY_{to}' for from_, to in enumerate(entity_order)}
            entity_order = {from_: to for from_, to in enumerate(entity_order)}

            rep = dict((re.escape(k), v) for k, v in itertools.chain(number_tokens.items(), entity_tokens.items())) 
            pattern = re.compile("|".join(rep.keys()))

            it['Question'] = pattern.sub(lambda m: rep[re.escape(m.group(0))], Question)
            it['ques source 1'] = pattern.sub(lambda m: rep[re.escape(m.group(0))], ques_source_1)
            it['Equation'] = pattern.sub(lambda m: rep[re.escape(m.group(0))], Equation)
            it['question'] = [pattern.sub(lambda m: rep[re.escape(m.group(0))], it) for it in question]
            it['equation'] = [pattern.sub(lambda m: rep[re.escape(m.group(0))], it) for it in equation]
            it['number list'] = [number_list[number_order[i]] for i in range(len(number_list))]
            it['entity list'] = [entity_list[entity_order[i]] for i in range(len(entity_list))]
            it['infix equation'] = [pattern.sub(lambda m: rep[re.escape(m.group(0))], it) for it in infix_equation]
            it['template'] = [pattern.sub(lambda m: rep[re.escape(m.group(0))], it) for it in template]
            
    def __init__(self, config):
        super().__init__(config)
        if config['tokenizer_path']:
            self.tokenizer = BertTokenizer.from_pretrained(config['tokenizer_path'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrained_model_path'])
        

        additional_mask_symbols = {self.mask_symbol, self.pre_mask}
        if MaskSymbol.NUM in additional_mask_symbols:
            self.tokenizer.add_special_tokens(dict(additional_special_tokens=['NUM']))
        if MaskSymbol.alphabet in additional_mask_symbols:
            self.tokenizer.add_special_tokens(dict(additional_special_tokens=NumMask.alphabet))
        if MaskSymbol.number in additional_mask_symbols:
            self.tokenizer.add_special_tokens(dict(additional_special_tokens=NumMask.number))
            
        if self.mask_entity:
            self.tokenizer.add_special_tokens(dict(additional_special_tokens=NumMask.entity))
            
        

        func_group_num_table = {
            'dep': get_group_num_by_dep,
            'pos': get_group_num_by_pos,
        }
        self.get_group_num = func_group_num_table[config['get_group_num']]

    def _preprocess(self):
        if self.mask_entity:
            for it in self.trainset:
                it['Question'], it['entity list'] = tag_entity(it['Question'])
            for it in self.validset:
                it['Question'], it['entity list'] = tag_entity(it['Question'])
            for it in self.testset:
                it['Question'], it['entity list'] = tag_entity(it['Question'])
                
        if self.dataset in ['kor_asdiv-a', DatasetName.hmwp]:
            self.trainset, self.validset, self.testset = id_reedit(self.trainset, self.validset, self.testset, id_key='ID')
        transfer = number_transfer_kor

        if self.mask_entity:
            self.trainset, generate_list, train_copy_nums, train_copy_etys, unk_symbol = transfer(self.trainset, self.dataset,
                                                                                 self.task_type, self.mask_symbol,
                                                                                 self.min_generate_keep, self.tokenizer,
                                                                                 self.pre_mask, self.mask_entity, ";")
            self.validset, _g, valid_copy_nums, valid_copy_etys, _ = transfer(self.validset, self.dataset, self.task_type, self.mask_symbol,
                                                             self.min_generate_keep, self.tokenizer, self.pre_mask, self.mask_entity, ";")
            self.testset, _g, test_copy_nums, test_copy_etys, _ = transfer(self.testset, self.dataset, self.task_type, self.mask_symbol,
                                                           self.min_generate_keep, self.tokenizer, self.pre_mask, self.mask_entity, ";")
        else:            
            self.trainset, generate_list, train_copy_nums, unk_symbol = transfer(self.trainset, self.dataset,
                                                                                 self.task_type, self.mask_symbol,
                                                                                 self.min_generate_keep, self.tokenizer,
                                                                                 self.pre_mask, self.mask_entity, ";")
            self.validset, _g, valid_copy_nums, _ = transfer(self.validset, self.dataset, self.task_type, self.mask_symbol,
                                                             self.min_generate_keep, self.tokenizer, self.pre_mask, self.mask_entity, ";")
            self.testset, _g, test_copy_nums, _ = transfer(self.testset, self.dataset, self.task_type, self.mask_symbol,
                                                           self.min_generate_keep, self.tokenizer, self.pre_mask, self.mask_entity, ";")

        target_equation_fix = self.equation_fix if self.equation_fix else FixType.Infix
        source_equation_fix = self.source_equation_fix if self.source_equation_fix else FixType.Infix
        if self.rule1:
            if source_equation_fix != FixType.Infix:
                warnings.warn("non-infix-equation datasets may not surport en rule1 process, already ignored it. ")
            elif self.linear and self.single:
                self.en_rule1_process(k=max([train_copy_nums, valid_copy_nums, test_copy_nums]))
            else:
                warnings.warn(
                    "non-linear or non-single datasets may not surport en rule1 process, already ignored it. ")
                # raise Warning("non-linear or non-single datasets may not surport en rule1 process, already ignored it. ")

        if self.rule2:
            if source_equation_fix != FixType.Infix:
                warnings.warn("non-infix-equation datasets may not surport en rule2 process, already ignored it. ")
            elif self.linear and self.single:
                self.en_rule2_process()
            else:
                warnings.warn(
                    "non-linear or non-single datasets may not surport en rule2 process, already ignored it. ")
                # raise Warning("non-linear or non-single datasets may not surport en rule2 process, already ignored it. ")

        if source_equation_fix == target_equation_fix:
            fix = None
        elif source_equation_fix == FixType.Infix and target_equation_fix == FixType.Prefix:
            fix = from_infix_to_prefix
        elif source_equation_fix == FixType.Infix and target_equation_fix == FixType.Postfix:
            fix = from_infix_to_postfix
        elif source_equation_fix == FixType.Prefix and target_equation_fix == FixType.Postfix:
            fix = from_prefix_to_postfix
        elif source_equation_fix == FixType.Prefix and target_equation_fix == FixType.Infix:
            fix = from_prefix_to_infix
        elif source_equation_fix == FixType.Postfix and target_equation_fix == FixType.Infix:
            fix = from_postfix_to_infix
        elif source_equation_fix == FixType.Postfix and target_equation_fix == FixType.Prefix:
            fix = from_postfix_to_prefix
        elif source_equation_fix == FixType.Infix and target_equation_fix == FixType.MultiWayTree:
            fix = from_infix_to_multi_way_tree
        else:
            raise NotImplementedError("the type of equation fix ({}) is not implemented.".format(self.equation_fix))

        self.fix_process(fix)
        self.operator_mask_process()

        self.generate_list = unk_symbol + generate_list
        if self.symbol_for_tree:
            self.copy_nums = max([train_copy_nums, valid_copy_nums, test_copy_nums])
            if self.mask_entity:
                self.copy_etys = max([train_copy_etys, valid_copy_etys, test_copy_etys])
        else:
            self.copy_nums = train_copy_nums
            if self.mask_entity:
                self.copy_etys = train_copy_etys
        if self.task_type == TaskType.SingleEquation:
            self.operator_list = copy.deepcopy(Operators.Single)
        elif self.task_type == TaskType.MultiEquation:
            self.operator_list = copy.deepcopy(Operators.Multi)
        self.operator_nums = len(self.operator_list)

        self.unk_symbol = unk_symbol

        # graph preprocess
        use_gpu = True if self.device == torch.device('cuda') else False
        # if self.model.lower() in ['graph2treeibm']:
        #     if os.path.exists(self.parse_tree_path) and not self.rebuild:
        #         logger = getLogger()
        #         logger.info("read deprel tree infomation from {} ...".format(self.parse_tree_path))
        #         self.trainset, self.validset, self.testset, token_list = \
        #             get_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)
        #     else:
        #         logger = getLogger()
        #         logger.info("build deprel tree infomation to {} ...".format(self.parse_tree_path))
        #         deprel_tree_to_file(self.trainset, self.validset, self.testset, \
        #                             self.parse_tree_path, self.language, use_gpu)
        #         self.trainset, self.validset, self.testset, token_list = \
        #             get_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)
        # if self.model.lower() in ['hms']:
        #     if os.path.exists(self.parse_tree_path) and not self.rebuild:
        #         logger = getLogger()
        #         logger.info("read span-level deprel tree infomation from {} ...".format(self.parse_tree_path))
        #         self.trainset, self.validset, self.testset, self.max_span_size = \
        #             get_span_level_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)
        #     else:
        #         logger = getLogger()
        #         logger.info("build span-level deprel tree infomation to {} ...".format(self.parse_tree_path))
        #         span_level_deprel_tree_to_file(self.trainset, self.validset, self.testset, \
        #                                        self.parse_tree_path, self.language, use_gpu)
        #         self.trainset, self.validset, self.testset, self.max_span_size = \
        #             get_span_level_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)
        # if self.model.lower() in ['graph2tree']:
        #     if os.path.exists(self.parse_tree_path) and not self.rebuild:
        #         logger = getLogger()
        #         logger.info("read deprel tree infomation from {} ...".format(self.parse_tree_path))
        #         self.trainset, self.validset, self.testset = \
        #             get_group_nums_(self.trainset, self.validset, self.testset, self.parse_tree_path)
        #     else:
        #         logger = getLogger()
        #         logger.info("build deprel tree infomation to {} ...".format(self.parse_tree_path))
        #         deprel_tree_to_file(self.trainset, self.validset, self.testset, \
        #                             self.parse_tree_path, self.language, use_gpu)
        #         self.trainset, self.validset, self.testset = \
        #             get_group_nums_(self.trainset, self.validset, self.testset, self.parse_tree_path)

        # if self.model.lower() in ['graph2tree']:
        #     logger = getLogger()
        #     logger.info("build kor deprel tree infomation to {} ...".format(self.parse_tree_path))
        #     q_infos = kor_deprel_tree_to_file(self.trainset, self.validset, self.testset, \
        #                             self.parse_tree_path, self.tokenizer, use_gpu)
        #     self.trainset, self.validset, self.testset = \
        #             kor_get_group_nums_(self.trainset, self.validset, self.testset, q_infos)


        if self.model.lower() in ['graph2tree']:
            if os.path.exists(self.parse_tree_path) and not self.rebuild:
                logger = getLogger()
                logger.info("read deprel tree infomation from {} ...".format(self.parse_tree_path))
                self.trainset, self.validset, self.testset = \
                    get_group_nums_kor(self.get_group_num, self.trainset, self.validset, self.testset, self.parse_tree_path)
            else:
                logger = getLogger()
                logger.info("build deprel tree infomation to {} ...".format(self.parse_tree_path))
                deprel_tree_to_file_kor(self.trainset, self.validset, self.testset, self.tokenizer, self.parse_tree_path)
                self.trainset, self.validset, self.testset = \
                    get_group_nums_kor(self.get_group_num, self.trainset, self.validset, self.testset, self.parse_tree_path)


    def _build_vocab(self):
        tokenizer = self.tokenizer

        if self.model.lower() in ['trnn']:
            tokenizer.add_tokens(self.generate_list)
        global SpecialTokens
        SpecialTokens.PAD_TOKEN=tokenizer.pad_token
        SpecialTokens.SOS_TOKEN=tokenizer.bos_token
        SpecialTokens.EOS_TOKEN=tokenizer.eos_token
        SpecialTokens.UNK_TOKEN=tokenizer.unk_token
        if self.embedding == 'bert':
            SpecialTokens.SOS_TOKEN=tokenizer.cls_token
            SpecialTokens.EOS_TOKEN=tokenizer.sep_token
        self.tokenizer=tokenizer
        self.in_idx2word = list(tokenizer.get_vocab().keys())

        if self.symbol_for_tree:
            self._build_symbol_for_tree()
            self._build_template_symbol_for_tree()
        elif self.equation_fix == FixType.MultiWayTree:
            self._build_symbol_for_multi_way_tree()
            self._build_template_symbol_for_multi_way_tree()
        else:
            self._build_symbol()
            self._build_template_symbol()

        self.in_word2idx = {}
        self.out_symbol2idx = {}
        self.temp_symbol2idx = {}
        for idx, word in enumerate(self.in_idx2word):
            self.in_word2idx[word] = idx
        for idx, symbol in enumerate(self.out_idx2symbol):
            self.out_symbol2idx[symbol] = idx
        for idx, symbol in enumerate(self.temp_idx2symbol):
            self.temp_symbol2idx[symbol] = idx


def number_transfer_kor(datas, dataset_name, task_type, mask_type, min_generate_keep, tokenizer, pre_mask, mask_entity, equ_split_symbol=';'):
    """number transfer

    Args:
        datas (list): dataset.
        dataset_name (str): dataset name.
        task_type (str): [single_equation | multi_equation], task type.
        min_generate_keep (int): generate number that count greater than the value, will be kept in output symbols.
        equ_split_symbol (str): equation split symbol, in multiple-equation dataset, symbol to split equations, this symbol will be repalced with special token SpecialTokens.BRG.
        tokenizer (BertTokenizer): tokenizer for korean text.
        pre_mask (str): mask symbol which was applied to input sequence before.

    Returns:
        tuple(list,list,int,list):
        processed datas, generate number list, copy number, unk symbol list.
    """
    transfer = _num_transfer_kor if pre_mask is not None else _num_transfer_transformer

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    if mask_entity:
        copy_etys = 0
    processed_datas = []
    unk_symbol = []
    for data in datas:
        if task_type == TaskType.SingleEquation:
            new_data = transfer(data, tokenizer, mask_type, pre_mask, mask_entity=mask_entity)
        elif task_type == TaskType.MultiEquation:
            new_data = transfer(data, tokenizer, mask_type, pre_mask, equ_split_symbol)
        else:
            raise NotImplementedError
        if dataset_name == DatasetName.mawps_single and task_type == TaskType.SingleEquation and '=' in new_data["equation"]:
            continue
        num_list = new_data["number list"]
        out_seq = new_data["equation"]
        copy_num = len(new_data["number list"])
        copy_ety = len(new_data["entity list"])

        for idx, s in enumerate(out_seq):
            # tag the num which is generated
            if s[0] == '-' and len(s) >= 2 and s[1].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        if copy_num > copy_nums:
            copy_nums = copy_num
            
        if mask_entity and copy_ety > copy_etys:
            copy_etys = copy_ety

        # get unknown number
        if task_type == TaskType.SingleEquation:
            pass
        elif task_type == TaskType.MultiEquation:
            for s in out_seq:
                if len(s) == 1 and s.isalpha():
                    if s in unk_symbol:
                        continue
                    else:
                        unk_symbol.append(s)
        else:
            raise NotImplementedError

        processed_datas.append(new_data)
    # keep generate number
    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    
    if mask_entity:
        return processed_datas, generate_number, copy_nums, copy_etys, unk_symbol
    return processed_datas, generate_number, copy_nums, unk_symbol


def _num_transfer_kor(data, tokenizer, mask_type, pre_mask, mask_entity=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    num_list = data['Numbers'].split() if pre_mask is not None else []
    seg = tokenizer.tokenize(data['Question'])
    equations = data['Equation']
    if equations.startswith('( ') and equations.endswith(' )'):
        equations = equations[2:-2]

    # match and split number
    input_seq = []
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            input_seq.append(str(str2float(s[pos.start():pos.end()])))
            if pos.end() < len(s):
                input_seq.append(s[pos.end():])
        else:
            input_seq.append(s)

    if pre_mask is not None:
        input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos_pre_masked(input_seq, num_list, mask_type, pre_mask)
    else:
        input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq, mask_type, pattern)

    if mask_entity:
        ety_list = data['entity list']
        # print(get_ety_pos_pre_masked(input_seq, ety_list))
        input_seq, ety_list, ety_pos, ety_all_pos, etys, ety_pos_dict, etys_for_ques = get_ety_pos_pre_masked(input_seq, ety_list)
        
    # out_seq = seg_and_tag_svamp(equations, nums_fraction, nums)
    out_seq = equations.split()

    source = copy.deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num

    if mask_entity:
        for pos in ety_all_pos:
            for key, value in ety_pos_dict.items():
                if pos in value:
                    ety_str = key
                    break
            source[pos] = ety_str
        
    source2 = tokenizer.convert_tokens_to_string(source)
    source = tokenizer.convert_tokens_to_string(input_seq)
    # source = ' '.join(source)
#     source = ' '.join(source).replace("#", "")

    new_data = data
    new_data["question"] = input_seq
    new_data["ques source 1"] = source
    new_data["ques source 2"] = source2     # for pororo
    new_data["equation"] = out_seq
    new_data["number list"] = num_list
    new_data["number position"] = num_pos
    if mask_entity:
        new_data["entity position"] = ety_pos
    new_data["id"] = str(data["ID"])
    new_data["ans"] = data["Answer"]
    return new_data


def sentence_preprocess(sentence):
    sentence = sentence.replace("(", "")
    sentence = sentence.replace(")", "")
    sentence = sentence.replace("$", "") # 이게 정보 일 수 있는데 제거 하는게 맞나
    sentence = sentence.replace("'", "")
    sentence = sentence.replace("`", "")
    return sentence


def transfer_digit_to_num(question):
    pattern = re.compile("\d+\/\d+%?|\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    nums = OrderedDict()
    num_list = []
    input_seq = []
    question = sentence_preprocess(question)
    seg = question.split(" ")
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            input_seq.append("NUM")
            num_list.append(str(str2float(s[pos.start():pos.end()])))
            if pos.end() < len(s):
                input_seq.append(s[pos.end():])
        else:
            input_seq.append(s)
    return " ".join(input_seq), num_list


def _num_transfer_transformer(data, tokenizer, mask_type, pre_mask='NUM'):

    question, num_list = transfer_digit_to_num(data['Question'])
    input_seq = tokenizer.tokenize(question)
    equations = data['Equation']
    if equations.startswith('( ') and equations.endswith(' )'):
        equations = equations[2:-2]

    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos_pre_masked(input_seq, num_list, mask_type, pre_mask)

    out_seq = seg_and_tag_svamp(equations, nums_fraction, nums)
    source = copy.deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = tokenizer.convert_tokens_to_string(source)

    new_data = data
    new_data["question"] = input_seq
    new_data["ques source 1"] = source
    new_data["equation"] = out_seq
    new_data["number list"] = num_list
    new_data["number position"] = num_pos
    new_data["id"] = str(data["ID"])
    new_data["ans"] = data["Answer"]
    return new_data

def get_ety_pos_pre_masked(input_seq, ety_list):
    sent_mask_list = NumMask.entity
    equ_mask_list = NumMask.entity
        
    pattern = re.compile(r'ETY_([0-9]+)')
    
    etys = OrderedDict()
    ety_pos = []
    ety_pos_dict = {}

    for word_pos, word in enumerate(input_seq):
        pos = re.search(pattern, word)
        if pos and pos.start() == 0:
            ety_idx = int(pos.groups()[0])
            word = ety_list[ety_idx]
            if word in ety_pos_dict:
                ety_pos_dict[word].append(word_pos)
            else:
                # num_list.append(word)
                ety_pos_dict[word] = [word_pos]
                
    # num_list = sorted(num_list, key=lambda x: max(num_pos_dict[x]), reverse=False)
    etys = lists2dict(ety_list, equ_mask_list[:len(ety_list)])

    etys_for_ques = lists2dict(ety_list, sent_mask_list[:len(ety_list)])

    # all number position
    all_pos = []
    
    for ety, mask in etys_for_ques.items():
        if ety in ety_pos_dict:
            for pos in ety_pos_dict[ety]:
                all_pos.append(pos)

    # final numbor position
    final_pos = []
    for ety in ety_list:
        if ety not in ety_pos_dict:
            continue
        final_pos.append(max(ety_pos_dict[ety]))

    # number transform
    for ety, mask in etys_for_ques.items():
        if ety in ety_pos_dict:
            for pos in ety_pos_dict[ety]:
                input_seq[pos] = mask
        else:
            print(ety, ety_list, ety_pos_dict, input_seq)

    return input_seq, ety_list, final_pos, all_pos, etys, ety_pos_dict, etys_for_ques

def get_num_pos_pre_masked(input_seq, num_list, mask_type, pre_mask):
    if pre_mask == MaskSymbol.NUM:
        pattern = re.compile('NUM')
    if pre_mask == MaskSymbol.alphabet:
        # pattern = re.compile('number_[A-Za-z]+')
        pattern = re.compile('NUM_[A-Za-z]+')
    if pre_mask == MaskSymbol.number:
        # pattern = re.compile('number[0-9]+')
        pattern = re.compile('NUM_([0-9]+)')

    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number
    nums = OrderedDict()
    num_pos = []
    num_pos_dict = {}

    if mask_type == MaskSymbol.NUM:
        # find all number position
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                num_idx = int(pos.groups()[0])
                # num_list.append(word)
                word = num_list[num_idx]
                num_pos.append(word_pos)
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_pos_dict[word] = [word_pos]

        mask_list = equ_mask_list[:len(num_list)]
        new_num_list = []
        new_mask_list = []
        for i in num_list:
            if num_list.count(i) != 1:
                x = 1
            if num_list.count(i) == 1:
                new_num_list.append(i)
                new_mask_list.append(mask_list[num_list.index(i)])
            else:
                pass
        nums = lists2dict(new_num_list, new_mask_list)
    else:
        # find all number position
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                num_idx = int(pos.groups()[0])
                word = num_list[num_idx]
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    # num_list.append(word)
                    num_pos_dict[word] = [word_pos]
        # num_list = sorted(num_list, key=lambda x: max(num_pos_dict[x]), reverse=False)
        nums = lists2dict(num_list, equ_mask_list[:len(num_list)])

    nums_for_ques = lists2dict(num_list, sent_mask_list[:len(num_list)])

    # all number position
    all_pos = []
    if mask_type == MaskSymbol.NUM:
        all_pos = copy.deepcopy(num_pos)
    else:
        for num, mask in nums_for_ques.items():
            if num in num_pos_dict:
                for pos in num_pos_dict[num]:
                    all_pos.append(pos)

    # final numbor position
    final_pos = []
    if mask_type == MaskSymbol.NUM:
        final_pos = copy.deepcopy(num_pos)
    else:
        for num in num_list:
            if num not in num_pos_dict:
                continue
            # select the latest position as the number position
            # if the number corresponds multiple positions
            final_pos.append(max(num_pos_dict[num]))

    # number transform
    for num, mask in nums_for_ques.items():
        if num in num_pos_dict:
            for pos in num_pos_dict[num]:
                input_seq[pos] = mask
        else:
            print(num, num_list, num_pos_dict, input_seq)

    nums_fraction = []
    for num, mask in nums.items():
        if re.search("\d*\(\d+/\d+\)\d*", num):
            nums_fraction.append(num)
    nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

    return input_seq, num_list, final_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction


def is_float_form(group, token):
    return (len(group) > 0 and str.isdecimal(group[-1][1]) and token == '.') or \
           (len(group) > 1 and str.isdecimal(group[-2][1]) and group[-1][1] == '.' and str.isdecimal(token))


def is_special_token(group, token):
    special_tokens = {'CLS', 'SEP', 'UNK', 'PAD', 'MASK'}
    return (token == '[') or \
           (len(group) > 0 and group[-1][1] == '[' and token in special_tokens) or \
           (len(group) > 1 and group[-2][1] == '[' and group[-1][1] in special_tokens and token == ']')


def is_num_token(group, token):
    return (token == 'NUM') or \
           (len(group) > 0 and group[-1][1] == 'NUM' and token == '_') or \
           (len(group) > 1 and group[-2][1] == 'NUM' and group[-1][1] == '_' and str.isdecimal(token))


def is_ety_token(group, token):
    return (token == 'ETY') or \
           (len(group) > 0 and group[-1][1] == 'ETY' and token == '_') or \
           (len(group) > 1 and group[-2][1] == 'ETY' and group[-1][1] == '_' and str.isdecimal(token))


def group_sub_tokens(tokens):
    token_group = []
    group = []
    for i, token in enumerate(tokens):
        if not token.startswith('##') and not is_float_form(group, token) and len(group) > 0:
            token_group.append(tuple(group))
            group = []
        group.append((i, token))
    if len(group) > 0:
        token_group.append(group)
    return token_group


def group_pos(pos_list):
    pos_group = []
    group = []
    for i, pos in enumerate(pos_list):
        t, p = pos
        if t.startswith('\u200b'):
            continue
        if p in {'SPACE', 'SC', 'SY', 'SF', 'SP', 'SSO', 'SSC', 'SE', 'SO'} \
                and not is_float_form(group, t) \
                and not is_special_token(group, t) \
                and not is_num_token(group, t) \
                and not is_ety_token(group, t):
            if len(group) > 0:
                pos_group.append(group)
            if p != 'SPACE':
                pos_group.append([(i,) + pos])
            group = []
            continue
        group.append((i,) + pos)
    if len(group) > 0:
        pos_group.append(group)
    return pos_group


def deprel_tree_to_file_kor(train_datas, valid_datas, test_datas, tokenizer, parse_tree_path):
    questions_infos = {}
    dp = Pororo(task="dep_parse", lang="ko")
    pos = Pororo(task='pos', lang='ko')
    # print("dataset length , ", len(train_datas), len(valid_datas), len(test_datas))
    questions_infos['trainset'] = get_token_info(train_datas, dp, pos, tokenizer)
    questions_infos['validset'] = get_token_info(valid_datas, dp, pos, tokenizer)
    questions_infos['testset'] = get_token_info(test_datas, dp, pos, tokenizer)

    write_json_data(questions_infos, parse_tree_path)


def get_token_info(dataset, dp, pos, tokenizer):
    questions_info = {}
    for data in dataset:
        question_list = []
        # question = tokenizer.convert_tokens_to_string(data["question"])
        # q, num_list = transfer_digit_to_num(question)   # input은 변경 가능
        tr = group_sub_tokens(data["question"])
        dpr = dp(sentence_preprocess_dp(data['ques source 2']))
        pr = group_pos(pos(data['ques source 1']))

        #잘못된 데이터 들어오면
        if len(tr) != len(dpr) or len(tr) != len(pr):
            print('grouping fail!')
            exit(1)
            if len(tr) != len(dpr):
                n = len(tr) - len(dpr)
                dpr += dpr[-n:]

            if len(tr) != len(pr):
                n = len(tr) - len(pr)
                pr += pr[-n:]

        for t_group, p_group, d_group in zip(tr, pr, dpr):
            for token in t_group:
                question_info = {
                    'token': token[1],
                    'token_pos': token[0],
                    'pos': p_group[0][2],
                    'deprel': d_group[3],
                    'head': d_group[2],
                    'dep_pos': d_group[0],
                }
                question_list.append(question_info)

        questions_info[str(data['id'])] = question_list

    return questions_info


def sentence_preprocess_dp(sentence):
    decimal = ['.', '?', '!', '~', ',', '`']
    result = ''
    for idx in range(len(sentence)):
        cur_char = sentence[idx]
        if idx > 0 and cur_char in decimal:
            pre_char = sentence[idx-1]

            if idx < len(sentence)-1:
                next_char = sentence[idx+1]
                if not next_char.isdigit():
                    result += " "
            elif not pre_char.isdigit():
                result += " "
        result += cur_char

    return result


def get_group_nums_kor(func_group_num, train_datas, valid_datas, test_datas, path):
    q_infos = read_json_data(path)
    trainset = func_group_num(train_datas, q_infos['trainset'])
    validset = func_group_num(valid_datas, q_infos['validset'])
    testset = func_group_num(test_datas, q_infos['testset'])
    return trainset, validset, testset


def get_group_num_by_pos(dataset, q_info):
    valid_tags = {
        'NNG', 'NNP', 'NNB', 'NNBC', 'NR', 'NP',  # nouns
        'MM', 'MAG',  # adjectives
        'VV', 'VA', 'VX', 'VCN', 'VCP',  # verbs, adjectives
        'SN',  # quantities
    }

    for data in dataset:
        question_id = str(data["id"])
        num_pos = data["number position"]
        group_nums = []
        info = q_info[question_id]

        sent_start_pos = [-1] + [i for i, x in enumerate(info) if x['pos'] == 'SF']
        sent_end_pos = [i for i, x in enumerate(info) if x['pos'] == 'SF'] + [len(info)]
        se_pos_set = {(s, e): False for s, e in zip(sent_start_pos, sent_end_pos)}
        for token_npos in num_pos:
            group_num = []
            start = max([x for x in sent_start_pos if x < token_npos])
            end = min([x for x in sent_end_pos if x > token_npos])
            se_pos_set[(start, end)] = True
            for token in info[start+1:end]:
                if not token['token'].startswith('##') and token['pos'] in valid_tags:
                    group_num.append(token['token_pos'])
            group_nums.append(group_num)
        common_group_num = []
        for (start, end), is_used in se_pos_set.items():
            if not is_used:
                for token in info[start+1:end]:
                    if not token['token'].startswith('##') and token['pos'] in valid_tags:
                        common_group_num.append(token['token_pos'])
        group_nums = [x + common_group_num for x in group_nums]
        data["group nums"] = group_nums
    return dataset


def get_group_num_by_dep(dataset, q_info):
    valid_tags = {
        'NNG', 'NNP', 'NNB', 'NNBC', 'NR', 'NP',  # nouns
        'MM', 'MAG',  # adjectives
        'VV', 'VA', 'VX', 'VCN', 'VCP',  # verbs, adjectives
        'SN',  # quantities
    }

    for data in dataset:
        question_id = str(data["id"])
        num_pos = data["number position"]
        group_nums = []
        info = q_info[question_id]

        dep_pos, dep_info, dep_head = get_dprel_info(info)
        sent_len = len(dep_pos)
        for token_npos in num_pos:
            npos = info[token_npos]['dep_pos']
            pos_stack = []
            group_num = []
            pos_stack.append(dep_head[npos-1])
            level = 0
            while pos_stack:
                head = pos_stack.pop(0)
                for token_pos in dep_pos[head-1]:
                    #and info[token_pos]["pos"] in valid_tags
                    if token_pos not in group_num and info[token_pos]["pos"] in valid_tags:
                        group_num.append(token_pos)

                # -1 최상위 , level -> depth 깊이
                if head != -1 and level < 2:
                    level += 1
                    for idx in range(len(dep_info)):
                        di = dep_info[idx]
                        #동일한 토큰 체크 and 같은 head를 가르키는 토큰 포함
                        if idx != head-1 and dep_head[idx] == head:
                            #현재 head 보다 전 토큰일땐, NP_SBJ만 선택
                            if idx < head and di != 'NP_SBJ':
                                continue

                            for token_pos in dep_pos[idx]:
                                #group_num에 토큰 없고 and valid_tags 있어야함
                                if token_pos not in group_num and info[token_pos]["pos"] in valid_tags:
                                    group_num.append(token_pos)
                    # next head
                    pos_stack.append(dep_head[head-1])
            if group_num == []:
                    group_num += dep_pos[npos-1]
            if npos - 1 >= 0:
                    group_num += dep_pos[npos-1]
            if npos + 1 < sent_len:
                    group_num += dep_pos[npos+1]
            group_nums.append(group_num)
        data["group nums"] = group_nums
    return dataset


def get_dprel_info(q_info):
    dep_pos = [ [] for _ in range(max([x['dep_pos'] for x in q_info]))]
    dep_info = [None] * max([x['dep_pos'] for x in q_info])
    dep_head = [None] * max([x['dep_pos'] for x in q_info])
    for idx, info in enumerate(q_info):
        dep_pos[info['dep_pos']-1].append(idx)
        dep_info[info['dep_pos']-1] = info['deprel']
        dep_head[info['dep_pos']-1] = info['head']
    return dep_pos, dep_info, dep_head


"""Deprecated"""

def pororo_pipeline(question, token_question, dp, pos, lemma, template_nlp):
    template_doc = template_nlp(token_question).to_dict()
    depenp = dp(question)
    # lemma_doc = lemma(question).to_dict()
    # pos_sentence = pos(question)
    # pos_sentence = [p for p in pos_sentence if p[1] != 'SPACE' ]

    depenp_idx = 0
    # pos_idx = 0
    #     lemma_idx = 0
    sentence_lemma_idx = 0
    current_word_for_depenp = ''
    # current_word_for_pos = ''
    # current_word_for_lemma = ''

    new_doc = []
    for sentence_idx in range(len(template_doc)):
        current_sentence = template_doc[sentence_idx]
        new_sentence = []

        for token in current_sentence:
            new_token = copy.deepcopy(token)
            current_word_for_depenp += token['text'].replace("#", "")
            #         current_word_for_pos += token['text'].replace("#","")
            #         current_word_for_lemma += token['text'].replace("#","")
            new_token['deprel'] = depenp[depenp_idx][3]
            #         new_token['lemma']=lemma_doc[sentence_lemma_idx][lemma_idx]['lemma']
            new_sentence.append(new_token)

            if current_word_for_depenp == depenp[depenp_idx][1] or len(current_word_for_depenp) > len(
                    depenp[depenp_idx][1]):
                current_word_for_depenp = ''
                depenp_idx += 1

        #         if current_word_for_pos == pos_sentence[pos_idx][0] or len(current_word_for_pos) > len(pos_sentence[pos_idx][0]):
        #             current_word_for_pos = ''
        #             pos_idx +=1

        #         if current_word_for_lemma == lemma_doc[sentence_lemma_idx][lemma_idx]['text'] or len(current_word_for_lemma) > len(lemma_doc[sentence_lemma_idx][lemma_idx]['text']):
        #             current_word_for_lemma = ''

        #             lemma_idx += 1
        #             if lemma_idx >= len(lemma_doc[sentence_lemma_idx]):
        #                 sentence_lemma_idx += 1
        #                 lemma_idx = 0
        new_doc.append(new_sentence)
    return new_doc


def kor_deprel_tree_to_file_(train_datas, valid_datas, test_datas, path, language, use_gpu):
    dp = Pororo(task="dep_parse", lang="ko")
    pos = Pororo(task="pos", lang="ko")
    lemma = stanza.Pipeline(lang='ko', processors='tokenize,lemma')
    template_nlp = stanza.Pipeline(lang='ko', processors='depparse,tokenize,pos,lemma', tokenize_pretokenized=True, logging_level='error', use_gpu=True)

    new_datas = []
    for idx, data in enumerate(train_datas):
        token_list = pororo_pipeline(data["ques source 2"], data["ques source 1"], dp, pos, lemma, template_nlp)
        new_datas.append({'id': data['id'], 'deprel': token_list})
    for idx, data in enumerate(valid_datas):
        token_list = pororo_pipeline(data["ques source 2"], data["ques source 1"], dp, pos, lemma, template_nlp)
        new_datas.append({'id': data['id'], 'deprel': token_list})
    for idx, data in enumerate(test_datas):
        token_list = pororo_pipeline(data["ques source 2"], data["ques source 1"], dp, pos, lemma, template_nlp)
        new_datas.append({'id': data['id'], 'deprel': token_list})
    write_json_data(new_datas, path)

    
    
def truncate_person_postfix(person):
    if person.endswith('이'):
        return person[:-1]
    return person

ner = Pororo(task='ner', lang='ko')
token_pattern = re.compile(r'[_A-Z0-9]')
def tag_entity(question):
    n = ner(question)
    
    res = []
    ignore_tag = ['O', 'QUANTITY', 'DATE', 'TERM', 'TIME']
    for w in n:
        if w[1] in ignore_tag:
            continue
            
        if w[1] == 'PERSON':
            w = (truncate_person_postfix(w[0]), w[1])
            
        if w[1] == 'ARTIFACT' and w[0].startswith('NUM'):
            continue
            
        if token_pattern.search(w[0]) is not None:
            continue
            
        if ' ' in w[0]:
            continue
            
        res.append(w[0])
            
    entities = sorted(list(set(res)), key=len, reverse=True)
    
    ret_entities = []
    pivot = 0
    for idx, entity in enumerate(entities):
        if entity in question:
            question = question.replace(entity, f'ETY_{pivot}')
            ret_entities.append(entity)
            pivot += 1
        
    return question, ret_entities

