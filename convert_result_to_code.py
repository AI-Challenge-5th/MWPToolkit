import string
import math
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict
import json

from io import StringIO
import sys

opset = {
    '+', '-', '*', '/', '%', '^',
    'sum', 'filter_eq', 'map_mod', 'range'
}


class VariableSpace:
    alphabets = '_' + string.ascii_uppercase

    def __init__(self):
        self.vars = {}
        self.count = 1

    def _get_next_varname(self):
        varname = ''

        N = len(self.alphabets)
        t = self.count
        while t > 0:
            r = t % N
            varname += self.alphabets[r]
            t //= N

        self.count += 1
        if self.count % N == 0:
            self.count += 1
        return varname[::-1]

    def add_var(self, code):
        varname = self._get_next_varname()
        self.vars[varname] = code
        return varname

    def to_code(self, print_var):
        result = 'import math\n\n'
        for varname, code in self.vars.items():
            result += f'{varname} = {code}\n'
        result += f'print(round({print_var}, 2))\n'
        return result, print_var


def parse_tree(tree_seq):
    if len(tree_seq) == 0:
        return None

    node = []
    cur = tree_seq[0]
    node.append(cur)

    if cur in opset:
        op = cur
        left, remain_seq = parse_tree(tree_seq[1:])
        node.append(left)

        right, remain_seq = parse_tree(remain_seq)
        node.append(right)

        return node, remain_seq

    else:
        return cur, tree_seq[1:]


def convert_node_to_code(node, var_space):
    head = node[0]
    # if first token is operand
    if head not in opset:
        return head

    # first token is operator
    op = head
    func = get_conversion_function(op)
    args = [x for x in node[1:]]
    code = func(*args)

    varname = var_space.add_var(code)
    return varname


def get_conversion_function(op):
    func_table = {
        '+': lambda x, y: f'({x} + {y})',
        '-': lambda x, y: f'({x} - {y})',
        '*': lambda x, y: f'({x} * {y})',
        '/': lambda x, y: f'({x} / {y})',
        '%': lambda x, y: f'({x} % {y})',
        '^': lambda x, y: f'({x} ** {y})',
        'range': lambda x, y: f'[x for x in range({x}, {y})]',
        'sum': lambda x, y: f'sum({x})',
        'factorial': lambda x: f'math.factorial({x})',
        'filter_eq': lambda x, y: f'[x for x, r in {x} if r == {y}]',
        'filter_neq': lambda x, y: f'[x for x, r in {x} if r != {y}]',
        'map_mod': lambda x, y: f'[(x, x % {y}) for x in {x}]',
    }
    if op not in func_table:
        return None
    return func_table[op]


def postfix_traverse(tree, node_func):
    if not isinstance(tree, list):
        return tree

    parent = tree[0]
    child_results = [postfix_traverse(child, node_func) for child in tree[1:]]

    result = node_func([parent] + child_results)
    return result


def convert_tree_to_code(tree, var_space):
    """ Convert equation in tree structure to Python code
    :argument
        tree: tree structural equation, result of `parse_tree()`
        number_dict: dict for feeding values into number placeholder, such like {"N1": 1, ...}
    :return
        string of generated Python code
    """

    return postfix_traverse(tree, lambda x: convert_node_to_code(x, var_space))


def convert_seq_to_code(seq):
    tree, _ = parse_tree(seq.split())
    vs = VariableSpace()
    final_varname = convert_tree_to_code(tree, vs)
    return vs.to_code(final_varname)


def read_result(path):
    with open(path, 'r') as f:
        results = json.load(f)
    results = [(x['id'], x['prediction'], x['number list']) for x in results]
    return results


def tree2code():
    test_set = read_result('outputs/result.json')
    answer_json = defaultdict(dict)
    for id, seq, num_list in test_set:
        try:
            code, print_var = convert_seq_to_code(seq)
            exec(code)
            if isinstance(locals()[print_var], float):
                locals()[print_var] = round(locals()[print_var], 2)
            answer_json[id]['answer'] = str(locals()[print_var])
            answer_json[id]['equation'] = code
        except Exception as e:
            answer_json[id]['answer'] = "0"
            answer_json[id]['equation'] = 'print(0)'
    with open('./answersheet_5_00_kesarr.json', 'w', encoding="utf-8") as f:
        json.dump(answer_json, f, indent=4)
    print('json file generated!')


def main():
    # test_set = [
    #     ('35', []),
    #     ('+ N0 N1', [9, 3]),
    #     ('/ N0 N1', [72, 8]),
    #     ('+ - N0 N1 N2', [100, 8, 15]),
    #     ('* - N0 N1 + N0 N1', [10, 3]),
    #     ('sum filter_eq map_mod range N0 N1 2 1 blank', [1, 200]),
    # ]
    # csv_file_path = os.path.join(config.outputs_path, config.dataset + '.csv')
    # test_set = csv2testset(csv_file_path)

    test_set = read_result('outputs/result.json')
    answer_json = defaultdict(dict)
    for id, seq, num_list in test_set:
        try:
            code, print_var = convert_seq_to_code(seq)
            print(f'ID: {id}')
            print(f'Equation: {seq}')
            print(f'```\n{code}```')
            print(f'Result: ', end='')
            exec(code)
            print('\n====')
                        
        except Exception as e:
            print(e)
            print(f'ID: {id}')
            print(f'Equation: {seq}')
            print(f'```\n print(1) \n```')
            print(f'Result: ', end='')
            exec('print(1)')
            print('\n====')


# +
# if __name__ == '__main__':
#     main()
