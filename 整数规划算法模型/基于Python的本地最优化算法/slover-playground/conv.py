#!/usr/bin/env python
#coding:utf-8

import sys
import os
import json
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
import pickle



def read_json(file_path):
    records = []

    with open(file_path, 'r') as reader:
        for line in tqdm(reader):
            record = json.loads(line.strip())
            new_record = {
                "all_ctcvr": [],
                "credit_ctcvr": [],
                "credit_activate": []
            }
            if "pre_action" not in record:
                continue
            pre_action = json.loads(record["pre_action"])
            if len(pre_action) < 4:
                continue
            for item in pre_action:
                if "credit_xtr" not in item:
                    continue
                credit_xtr = json.loads(item['credit_xtr'])
                new_record["all_ctcvr"].append(credit_xtr['all_ctcvr'])
                new_record["credit_ctcvr"].append(credit_xtr['credit_ctcvr'])
                new_record["credit_activate"].append(credit_xtr['credit_activate'])

            records.append(new_record)

    return records


def main():
    records = read_json(sys.argv[1])
    with open(sys.argv[2], 'w') as writer:
        json.dump(records, writer)

if __name__ == "__main__":
    main()