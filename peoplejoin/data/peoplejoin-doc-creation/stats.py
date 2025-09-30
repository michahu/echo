#!/usr/bin/env python3

import os
import json
import numpy as np

ndocs = []
nusers = []
nuserdocs = []
nalldocs = []

gold_lens = []
gold_users = []

with open("test.scenario.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        ndocs.append(len(doc["gold_document_ids"]))
        tenant = doc['datum_id'].rsplit("_", 1)[0]
        gold_len = 0
        gold_user = 0
        with open(f"tenants/{tenant}.json") as f2:
            tdoc = json.load(f2)
            for artset in tdoc["user_id_to_documents"].values():
                in_gold = False
                for art in artset:
                    if art["title"] not in doc["gold_document_ids"]:
                        continue
                    in_gold = True
                    gold_len += len(art["content"].split())
                gold_user += int(in_gold)
        gold_lens.append(gold_len)
        gold_users.append(gold_user)

for name in os.listdir("tenants"):
    if not name.startswith("test_"):
        continue
    with open(f"tenants/{name}") as f:
        doc = json.load(f)
        nusers.append(len(doc["users"]))
        alldocs = 0
        for k, v in doc["user_id_to_documents"].items():
            nuserdocs.append(len(v))
            alldocs += len(v)
        nalldocs.append(alldocs)


print("ndocs", np.mean(ndocs), np.var(ndocs))
print("nusers", np.mean(nusers), np.var(nusers))
print("nuserdocs", np.mean(nuserdocs), np.var(nuserdocs))
print("nalldocs", np.mean(nalldocs), np.var(nalldocs))
print("gold_len", np.mean(gold_lens), np.var(gold_lens))
print("gold_user", np.mean(gold_users), np.var(gold_users))
