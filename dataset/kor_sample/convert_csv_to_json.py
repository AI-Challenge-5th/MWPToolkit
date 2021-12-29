import pandas as pd
import csv
import json


def place_numbers(text, numbers):
    for i, num in enumerate(numbers.split()):
        text = text.replace(f'number{i}', num)
    return text


def main():
    src_file_names = ['train.csv', 'dev.csv', 'dev.csv']
    tgt_file_names = ['trainset.json', 'validset.json', 'testset.json']

    for src, tgt in zip(src_file_names, tgt_file_names):
        c = pd.read_csv(src, encoding='utf-8').reset_index().rename(columns={'index': 'ID'})
        for col in ['Question', 'Equation', 'Body', 'Ques']:
            c[col] = c.apply(lambda row: place_numbers(row[col], row['Numbers']), axis=1)
        d = c.to_json(None, orient='records', force_ascii=False, indent=4)
        with open(tgt, 'w', encoding='utf-8') as f:
            f.write(d)


if __name__ == '__main__':
    main()
