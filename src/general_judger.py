#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region ------------ LIBRARIES ------------------- #
import json, argparse
from pprint import pprint; from tqdm import tqdm
from collections import OrderedDict

import numpy as np, pandas as pd
from IPython.display import display, HTML
# endregion
# region ------------ ARGUMENTS ------------------- #
args = argparse.ArgumentParser()
args.add_argument("COOL_FILE", 
                # default="HW15 測驗 學生分析 報告.csv", 
                  help="File name of downloaded "
                       "COOL \033[33mCSV\033[0m", 
                  type=str)
args.add_argument("OUTPUT_FILE", 
                  help="Output \033[33mCSV\033[0m "
                       "file name", 
                  type=str)
args.add_argument("STANDARD_SOLUTION", nargs="?",
                  default="standard.json", 
                  help="Standard solution file name "
                       "(.\033[33mjson\033[0m)",
                  type=str)
args = args.parse_args()
# endregion
# region ------------ CONSTANTS ------------------- #
NAN = "NAN"
HEADER_EN = ['name',      'id',         'sis_id', 
            'section', 'section_id', 'section_sis_id', 
             'submitted', 'attempt']
HEADER_ZH = ['名稱',   'ID',         'sis_id',
             '班別',   'section_id', 'section_sis_id',
             '已提交', '作答紀錄']
# endregion
# region -------------- UTILS --------------------- #
def set_equal(inp: str, ref: str) -> bool:
    """ 
    Check if the input is the same as the reference. 
    """
    global NAN
    if inp is NAN: return False
    if isinstance(ref, list):
        return bool(sum(
            ([set_equal(inp, possible_ref) 
                for possible_ref in ref])))
    inp_comma, ref_comma = \
        inp.replace("\\,", " <<<COMMA>>> "), \
        ref.replace("\\,", " <<<COMMA>>> ")
    return set(inp_comma.split(',')) \
        == set(ref_comma.split(','))
def deduplicator(res, method="latest"):
    """
    Given a data frame, deduplicate by `method'.
    """
    dedup_dict = OrderedDict()
    dedup_dict_best = OrderedDict()

    if method == "latest":
        # > ---      Method 1: The LATEST      --- < #
        # reverse, keep latest
        for it in np.array(res.values).copy()[::-1]:
            dedup_dict[tuple(it[:5])] = it
    elif method == "best":
        # > ---       Method 2: The BEST       --- < #
        # reverse, keep latest
        # @ from old to new
        for it in np.array(res.values).copy()[::-1]:
            person = tuple(it[:5])
            if person in dedup_dict:
                assert (person in dedup_dict_best
                    ), "Two dictionary not synced!"
                # Get a better grade
                if it[-1] >= dedup_dict_best[person]:
                    dedup_dict[person] = it
                    dedup_dict_best[person] = it[-1]
            else:  # first attempt
                dedup_dict[person] = it
                dedup_dict_best[person] = it[-1]
    # reverse back
    final_results = pd.DataFrame(
        np.array([dedup_dict[i] 
            for i in dedup_dict][::-1]), 
        columns=res.columns)
    return final_results
# endregion
# region -------------- PROCESS -------------------- #
# > ---              Read in data              --- < #
data = pd.read_csv(args.COOL_FILE).fillna(NAN)
columns = data.columns.tolist()

# > ---             Check language             --- < #
HEADER = (     HEADER_ZH if columns[:8] == HEADER_ZH 
          else HEADER_EN)
assert columns[:8] == HEADER, "Header wrong!"        

# > - Filter out statements, keep real problems -- < #
content_columns = columns[8:]                         
problems = [prob for prob in content_columns          
                 if prob.split()[0][-1] == ":"]       
problems = [prob for prob in problems            
                 if set(data[prob]) != {NAN}]    
print(  # ( Print out the results ) #
    f"We have {len(problems)} problems in the quiz!")

# > ---       Find the reference answer        --- < #
if "HW15" in args.COOL_FILE:
    if HEADER == HEADER_ZH:
        REF_NAME = {"名稱": "助教群"}
    elif HEADER == HEADER_EN:
        REF_NAME = {"name": "助教群"}
    else:
        assert False, "Unknown language!"
elif "HW14" in args.COOL_FILE:
    if HEADER == HEADER_ZH:
        REF_NAME = {
            "名稱": "陳建成 (CHEN, CHIEN-CHENG)",
            "ID": 75454,
            "作答紀錄": 2
        }
    elif HEADER == HEADER_EN:
        REF_NAME = {
            "name": "陳建成 (CHEN, CHIEN-CHENG)",
            "ID": 75454,
            "attempt": 2
        }
    else:
        assert False, "Unknown language!"
else:
    raise NotImplementedError

TA_answer = data.copy()
for col, val in REF_NAME.items():
    TA_answer = TA_answer.loc[TA_answer[col] == val]
assert not TA_answer.empty, "No reference user!"
TA_answer = TA_answer[problems].values.tolist()
assert len(TA_answer) == 1, "More than 1 reference!"
TA_answer = TA_answer[0]
# endregion
# region -------------- ADJUST -------------------- #
if "HW15" in args.COOL_FILE:
    optionA = ("For a meta-batch, MAML updates the mo"
        "del parameters in the outer-loop after it ha"
        "s finished all the inner-loop steps. But wit"
        "h multi-Step loss optimization, it minimizes"
        " the target set loss computed by the base-ne"
        "twork after every step towards a support set"
        " task.").replace(',', "\\,")
    optionB = ("Learning a learning rate and directio"
        "n for each layer in the network is better th"
        "an learning a learning rate and gradient dir"
        "ection for each parameter in the base-networ"
        "k since the latter causes increased number o"
        "f parameters and increased computational ove"
        "rhead.").replace(',', "\\,")
    optionD = ("Learning a set of biases per-step wit"
        "hin the inner-loop update process may help f"
        "ix the shared (across step) batch normalizat"
        "ion bias problem of the original MAML traini"
        "ng method.").replace(',', "\\,")
    
    TA_answer[22 - 1] = [','.join(combi) for combi in [
        # original answer
        [optionA, optionB],
        # other answers
        [optionA, optionD],
        [optionB, optionD],
        [optionA, optionB, optionD]]]

elif "HW14" in args.COOL_FILE:
    # TA_answer[3 - 1] = [
    #     '『多任務學習是希望一個模型可以處理很多不同的任務，所以在訓練時，會將所有任務的訓練資料倒在一起，變成一個巨量的訓練資料，一次使用多個任務的訓練資料以及對應的目標函數一起更新模型的參數。』 “Multi-task learning hopes that a model can handle many different tasks. Therefore, during training, the training data of all tasks will be poured together to become a huge amount of training data. Using the training data of multiple tasks and the corresponding objective functions to update the parameters of the model at a time.”',
    #     '『終身機器學習是希望一個模型可以處理不同的任務，為了做到這件事情，會希望模型經過訓練後，學習到一個好的參數起始點，之後，透過極少訓練資料以及更新次數，適應在新的任務上。』 “Life long learning hopes that a model can handle different tasks. To do this, we hoped that the model would learn a good starting point for parameters after training. After that, it will adapt to the new task through very little training data and update times.”',
    # ]
    pass
else:
    raise NotImplementedError
# endregion
# region -------- DUMP THE REFERENCE -------------- #
standard_QA_pairs = \
    OrderedDict(i for i in zip(problems, TA_answer))
with open(args.STANDARD_SOLUTION, 'w') as f:
    json.dump(standard_QA_pairs, f, 
              ensure_ascii=False, indent=4)
# endregion
# region ----------- READ IN DATA ------------------ #
data = data[HEADER + problems]
res = pd.DataFrame()
res[HEADER] = data[HEADER]
QUES_NUM = []
for en, prb in enumerate(problems):
    ans = TA_answer[en]
    check_judge = {
        i: (0.4 if set_equal(i, ans) else 0.) 
        for i in set(data[prb])}
    print(f"Prob. {en + 1:2d}")
    print(prb)
    print(ans)
    pprint(check_judge)
    print()
    res[str(en + 1)] = data[prb].map(check_judge)
    QUES_NUM.append(str(en + 1))

res['total'] = (res.loc[:, QUES_NUM]
                   .sum(axis=1).round(2))
# endregion
# region ---------- DEDUPLICATION ----------------- #
res.to_csv("original__" + args.OUTPUT_FILE)
deduplicator(res).to_csv(args.OUTPUT_FILE)
deduplicator(res, "best").to_csv(
    "best__" + args.OUTPUT_FILE)
# endregion
