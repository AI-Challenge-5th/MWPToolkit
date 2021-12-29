# -*- coding: utf-8 -*-
import json
import argparse
from mwptoolkit.data.dataset.korean_dataset import transfer_digit_to_str
import pathlib
from pororo import Pororo
import re

DATA_PATH = '/home/agc2021/dataset/problemsheet.json'
EVAL_PATH ='./dataset/eval/'

# +
def ner_quantity(q, tk):
#     tk = Pororo(task='ner', lang='ko')
    cnt = 0
    numbers = []
    q_ = ''
    for i in tk(q):
        if i[1] == 'QUANTITY':
            prior_end = 0
            for p in re.finditer(r'((-?[0-9]+(?:,[0-9]{3})*(\.[0-9]+| ?/ ?-?[0-9]+)?)|(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열) ?(개|칸|마리|권|번|자리|명|가지|사람)|(첫|둘|셋|넷) ?(번|째))', i[0]):
                if p.group(2):
                    numbers.append(p.group(2))
                    q_ += i[0][prior_end:p.start()]
                    q_ += f' NUM_{cnt} '
                    cnt += 1
                    prior_end = p.end()
                elif p.group(4):
                    numbers.append(p.group(4))
                    q_ += i[0][prior_end:p.start()]
                    q_ += f' NUM_{cnt} '
                    cnt += 1
                    q_ += p.group(5)
                    prior_end = p.end()
                elif p.group(6):
                    numbers.append(p.group(6))
                    q_ += i[0][prior_end:p.start()]
                    q_ += f' NUM_{cnt} '
                    cnt += 1
                    q_ += p.group(7)
                    prior_end = p.end()
                else:
                    # quantity라고 하는데 number token으로 masking할 문자열을 찾지 못한 경우.
                    pass
            q_ += i[0][prior_end:]
        else:
            q_ += i[0]
            
    new_numbers = []
    for i in numbers:
        if (i == '한') or (i == '첫'):
            new_numbers.append('1')
        elif (i == '두') or (i == '둘'):
            new_numbers.append('2')
        elif (i == '세') or (i == '셋'):
            new_numbers.append('3')
        elif (i == '네') or (i == '넷'):
            new_numbers.append('4')
        elif (i == '다섯'):
            new_numbers.append('5')
        elif (i == '여섯'):
            new_numbers.append('6')
        elif (i == '일곱'):
            new_numbers.append('7')
        elif (i == '여덟'):
            new_numbers.append('8')
        elif (i == '아홉'):
            new_numbers.append('9')
        elif (i == '열'):
            new_numbers.append('10')
        else:
            new_numbers.append(i.replace(',',''))
    
    if not new_numbers:
        new_numbers = ['0']
        q_ += ' NUM_0'
       
    return re.sub("\s+" , " ", q_), new_numbers


# q = '어떤 소수의 소수점을 오른쪽으로 한자리 옮기면 원래보다 2.7만큼 커집니다. 원래의 소수를 구하시오.'
# q1 = '어떤 소수의 소수점을 오른쪽으로 한 자리 옮기면 원래보다 2.7/234만큼 커집니다. 원래의 소수를 구하시오.'
# q2 = '5개의 수 1.4, 9/10, 1.1, 0.5, 13/10이 있습니다. 이 중에서 0.9보다 큰 수는 모두 몇 개입니까?'
# q3 = '5,000부터 1,050,000까지의 수 중에서 2,000원 배수가 아닌 두 사람들의 합을 구하시오 첫 번째.'
# tk = Pororo(task='ner', lang='ko')
# a,b = ner_quantity(q, tk)
# print(a,b)
# a,b = ner_quantity(q1, tk)
# print(a,b)
# a,b = ner_quantity(q2, tk)
# print(a,b)
# a,b = ner_quantity(q3, tk)
# print(a,b)
# 어떤 소수의 소수점을 오른쪽으로 number0 자리 옮기면 원래보다 number1 만큼 커집니다. 원래의 소수를 구하시오. ['한', '2.7']
# 어떤 소수의 소수점을 오른쪽으로 number0 자리 옮기면 원래보다 number1 만큼 커집니다. 원래의 소수를 구하시오. ['한', '2.7']
# number0 개의 수 number1 , number2 , number3 , number4 , number5 이 있습니다. 이 중에서 number6 보다 큰 수는 모두 몇 개입니까? ['5', '1.4', '9/10', '1.1', '0.5', '13/10', '0.9']

# +
def sheet2json_main(args):
    data_path = args.data_path
    eval_path = args.eval_path
    pathlib.Path(eval_path).mkdir(parents=True, exist_ok=True) 
    with open(data_path, encoding='utf-8-sig') as f:
        data = json.load(f)
        
    total_question_length = len(data)+1
    tk = Pororo(task='ner', lang='ko')
    problem_list = []
    for i in range(1, total_question_length):
        q_dict = {}

#         mask_question, num_list = transfer_digit_to_str(data[str(i)]['question'])
        mask_question, num_list = ner_quantity(data[str(i)]['question'], tk)

        q_dict['Question'] = mask_question
        q_dict['Numbers'] = " ".join(num_list)
        q_dict['Answer'] = 1
        q_dict['Equation'] = "- NUM_0 NUM_1"
        q_dict['ID']=str(i)
        problem_list.append(q_dict)

    with open(eval_path+'testset.json', 'w', encoding='utf-8-sig') as f:
        json.dump(problem_list, f,  indent="\t")
    with open(eval_path+'trainset.json', 'w', encoding='utf-8-sig') as f:
        json.dump([], f,  indent="\t")
    with open(eval_path+'validset.json', 'w', encoding='utf-8-sig') as f:
        json.dump([], f,  indent="\t")
