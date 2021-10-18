# -*- encoding: utf-8 -*-
# @Author: Junwon Hwang
# @Time: --
# @File: korean_dataset.py


import os
import copy
import warnings
from logging import getLogger
import re
from collections import OrderedDict

import torch
from transformers import RobertaTokenizer,BertTokenizer,AutoTokenizer

from mwptoolkit.data.dataset.abstract_dataset import AbstractDataset
from mwptoolkit.data.dataset.pretrain_dataset import PretrainDataset
from mwptoolkit.utils.enum_type import DatasetName, MaskSymbol, NumMask,TaskType,FixType,Operators,SpecialTokens
from mwptoolkit.utils.preprocess_tools import id_reedit
from mwptoolkit.utils.preprocess_tool.equation_operator import from_infix_to_multi_way_tree
from mwptoolkit.utils.preprocess_tool.equation_operator import from_infix_to_postfix, from_infix_to_prefix, from_postfix_to_infix, from_postfix_to_prefix, from_prefix_to_infix, from_prefix_to_postfix
from mwptoolkit.utils.preprocess_tool.sentence_operator import deprel_tree_to_file, get_group_nums_, span_level_deprel_tree_to_file, get_span_level_deprel_tree_, get_deprel_tree_
from mwptoolkit.utils.preprocess_tool.number_transfer import seg_and_tag_svamp, get_num_pos
from mwptoolkit.utils.utils import read_json_data, str2float


class KoreanRobertaDataset(PretrainDataset):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = BertTokenizer.from_pretrained(config['pretrained_model_path'])

    def _preprocess(self):
        if self.dataset in [DatasetName.hmwp]:
            self.trainset, self.validset, self.testset = id_reedit(self.trainset, self.validset, self.testset)
        transfer = number_transfer_kor

        self.trainset, generate_list, train_copy_nums, unk_symbol = transfer(self.trainset, self.dataset,
                                                                             self.task_type, self.mask_symbol,
                                                                             self.min_generate_keep, self.tokenizer, ";")
        self.validset, _g, valid_copy_nums, _ = transfer(self.validset, self.dataset, self.task_type, self.mask_symbol,
                                                         self.min_generate_keep, self.tokenizer, ";")
        self.testset, _g, test_copy_nums, _ = transfer(self.testset, self.dataset, self.task_type, self.mask_symbol,
                                                       self.min_generate_keep, self.tokenizer, ";")

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
        else:
            self.copy_nums = train_copy_nums
        if self.task_type == TaskType.SingleEquation:
            self.operator_list = copy.deepcopy(Operators.Single)
        elif self.task_type == TaskType.MultiEquation:
            self.operator_list = copy.deepcopy(Operators.Multi)
        self.operator_nums = len(self.operator_list)

        self.unk_symbol = unk_symbol

        # graph preprocess
        use_gpu = True if self.device == torch.device('cuda') else False
        if self.model.lower() in ['graph2treeibm']:
            if os.path.exists(self.parse_tree_path) and not self.rebuild:
                logger = getLogger()
                logger.info("read deprel tree infomation from {} ...".format(self.parse_tree_path))
                self.trainset, self.validset, self.testset, token_list = \
                    get_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)
            else:
                logger = getLogger()
                logger.info("build deprel tree infomation to {} ...".format(self.parse_tree_path))
                deprel_tree_to_file(self.trainset, self.validset, self.testset, \
                                    self.parse_tree_path, self.language, use_gpu)
                self.trainset, self.validset, self.testset, token_list = \
                    get_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)
        if self.model.lower() in ['hms']:
            if os.path.exists(self.parse_tree_path) and not self.rebuild:
                logger = getLogger()
                logger.info("read span-level deprel tree infomation from {} ...".format(self.parse_tree_path))
                self.trainset, self.validset, self.testset, self.max_span_size = \
                    get_span_level_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)
            else:
                logger = getLogger()
                logger.info("build span-level deprel tree infomation to {} ...".format(self.parse_tree_path))
                span_level_deprel_tree_to_file(self.trainset, self.validset, self.testset, \
                                               self.parse_tree_path, self.language, use_gpu)
                self.trainset, self.validset, self.testset, self.max_span_size = \
                    get_span_level_deprel_tree_(self.trainset, self.validset, self.testset, self.parse_tree_path)
        if self.model.lower() in ['graph2tree']:
            if os.path.exists(self.parse_tree_path) and not self.rebuild:
                logger = getLogger()
                logger.info("read deprel tree infomation from {} ...".format(self.parse_tree_path))
                self.trainset, self.validset, self.testset = \
                    get_group_nums_(self.trainset, self.validset, self.testset, self.parse_tree_path)
            else:
                logger = getLogger()
                logger.info("build deprel tree infomation to {} ...".format(self.parse_tree_path))
                deprel_tree_to_file(self.trainset, self.validset, self.testset, \
                                    self.parse_tree_path, self.language, use_gpu)
                self.trainset, self.validset, self.testset = \
                    get_group_nums_(self.trainset, self.validset, self.testset, self.parse_tree_path)


def number_transfer_kor(datas, dataset_name, task_type, mask_type, min_generate_keep, tokenizer, equ_split_symbol=';'):
    """number transfer

    Args:
        datas (list): dataset.
        dataset_name (str): dataset name.
        task_type (str): [single_equation | multi_equation], task type.
        min_generate_keep (int): generate number that count greater than the value, will be kept in output symbols.
        equ_split_symbol (str): equation split symbol, in multiple-equation dataset, symbol to split equations, this symbol will be repalced with special token SpecialTokens.BRG.

    Returns:
        tuple(list,list,int,list):
        processed datas, generate number list, copy number, unk symbol list.
    """
    transfer = _num_transfer_kor

    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    unk_symbol = []
    for data in datas:
        if task_type == TaskType.SingleEquation:
            new_data = transfer(data, tokenizer, mask_type)
        elif task_type == TaskType.MultiEquation:
            new_data = transfer(data, tokenizer, mask_type, equ_split_symbol)
        else:
            raise NotImplementedError
        if dataset_name == DatasetName.mawps_single and task_type == TaskType.SingleEquation and '=' in new_data["equation"]:
            continue
        num_list = new_data["number list"]
        out_seq = new_data["equation"]
        copy_num = len(new_data["number list"])

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
    return processed_datas, generate_number, copy_nums, unk_symbol


def _num_transfer_kor(data, tokenizer, mask_type):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")

    nums = OrderedDict()
    num_list = []
    input_seq = []
    seg = tokenizer.tokenize(data['Question'])
    equations = data['Equation']
    if equations.startswith('( ') and equations.endswith(' )'):
        equations = equations[2:-2]

    num_pos_dict = {}
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

    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq, mask_type, pattern)

    out_seq = seg_and_tag_svamp(equations, nums_fraction, nums)

    source = copy.deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    new_data = data
    new_data["question"] = input_seq
    new_data["ques source 1"] = source
    new_data["equation"] = out_seq
    new_data["number list"] = num_list
    new_data["number position"] = num_pos
    new_data["id"] = data["ID"]
    new_data["ans"] = data["Answer"]

    return new_data
