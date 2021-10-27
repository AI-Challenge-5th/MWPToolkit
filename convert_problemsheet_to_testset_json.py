import json
import argparse
from mwptoolkit.data.dataset.korean_dataset import transfer_digit_to_str

DATA_PATH = './agc2021/dataset/problemsheet.json'
EVAL_PATH ='./dataset/eval/'

def main(args):
    data_path = args.data_path
    eval_path = args.eval_path

    with open(data_path, encoding='utf-8-sig') as f:
        data = json.load(f)
        
    total_question_length = len(data)+1
    
    problem_list = []
    for i in range(1, total_question_length):
        q_dict = {}
        mask_question, num_list = transfer_digit_to_str(data[str(i)]['question'])
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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='convert eval')
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    parser.add_argument('--eval_path', type=str, default=EVAL_PATH)
    args = parser.parse_args()
    main(args)

