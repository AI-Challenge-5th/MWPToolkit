from convert_problemsheet_to_testset_json import sheet2json_main
from convert_result_to_code import tree2code as res2code_main
from easydict import EasyDict
import sys
import os

from mwptoolkit.quick_start import run_toolkit

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))


def get_sheet2json_config():
    return EasyDict({"data_path":"./agc2021/dataset/problemsheet.json",
              "eval_path":"./dataset/eval/"})


def get_mwptoolkit_config():
    return EasyDict({"model":"Graph2Tree",
                     "dataset":"eval",
                     "task_type":"single_equation",
                     "gpu_id":0,
                     "equation_fix":"prefix",
                     "embedding":"koelectra",
                     "pretrained_model_path":"./test_trainer/5_epoch_1000",
                     "tokenizer_path":"monologg/koelectra-base-v3-discriminator",
                     "add_sos":False,
                     "add_eos":False,
                     "embedding_size":768,
                     "mask_symbol":"number",
                     "pre_mask":"number",
                     "test_only":True,
                     "output_path":"outputs/result.json",
                     "get_group_num":"pos",
                     "encode_type":"seg",
                     "rebuild":True,
                     "prompt":True,})


if __name__=="__main__":
    
    sheet2json_args = get_sheet2json_config()
    sheet2json_main(sheet2json_args)
    toolkit_args = get_mwptoolkit_config()
    run_toolkit(toolkit_args.model, toolkit_args.dataset, toolkit_args.task_type, toolkit_args)
    res2code_main()

