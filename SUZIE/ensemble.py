import os
import json
import pandas as pd
from collections import Counter

submission_dir = './submission'


if __name__ == '__main__':
    submission_list = os.listdir(submission_dir)
    submissions = []
    for submission_name in submission_list:
        if submission_name == "sample_submission.json":
            continue
        submissions.append(pd.read_json(os.path.join(submission_dir, submission_name)))

    ID = list(submissions[0]['ID'])
    indices = []
    for submission in submissions:
        indices.append(list(submission['summary_index1']))
        indices.append(list(submission['summary_index2']))
        indices.append(list(submission['summary_index3']))

    count_dict = {}
    for i in range(len(indices[0])):
        row = []
        for index in indices:
            row.append(index[i])
        count_dict[ID[i]] = [tup[0] for tup in Counter(row).most_common()][:3]

    pred_lst = list(count_dict.values())

    with open(os.path.join(submission_dir, "sample_submission.json"), "r", encoding="utf-8-sig") as f:
        final_submission = json.load(f)

    for row, pred in zip(final_submission, pred_lst):
        row['summary_index1'] = pred[0]
        row['summary_index2'] = pred[1]
        row['summary_index3'] = pred[2]

    with open(os.path.join(submission_dir, "final_submission.json"), "w") as f:
        json.dump(final_submission, f, separators=(',', ':'))