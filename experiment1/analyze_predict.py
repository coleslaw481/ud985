#!/usr/bin/env python

import json
import sys

if len(sys.argv) != 2:
    sys.stderr.write('\nUsage <aggregated prediction json output>\n')
    sys.stderr.write('Simple script that takes '
                     'aggregated output of '
                     'predict_clusters.py json output to get summary stats\n\n')
    sys.exit(1)
with open(sys.argv[1], 'r') as f:
    exact_cnt = 0
    one_percent_cnt = 0
    five_percent_cnt = 0
    ten_percent_cnt = 0
    twenty_percent_cnt = 0
    counter = 0
    for line in f:
        predict_json = json.loads(line)
        counter += 1
        # print(str(predict_json['predictedNumberOfClusters']) +
        #      ' vs ' + str(predict_json['actualNumberOfClusters']))
        predict_clusters = predict_json['predictedNumberOfClusters']
        actual_clusters = predict_json['actualNumberOfClusters']
        if predict_clusters == actual_clusters:
            exact_cnt += 1
            continue
        abs_diff = abs(predict_clusters - actual_clusters)
        pc_diff = (float(abs_diff) / float(actual_clusters))
        if pc_diff <= 0.01:
            one_percent_cnt += 1
        elif pc_diff <= 0.05:
            five_percent_cnt += 1
        elif pc_diff <= 0.1:
            ten_percent_cnt += 1
        elif pc_diff <= 0.2:
            twenty_percent_cnt += 1

print('Percent exact matches: ' + str(round((exact_cnt / counter)*100, 1)) + '%')
print('Percent within 1% : ' + str(round(((one_percent_cnt + exact_cnt) / counter)*100, 1)) + '%')
print('Percent within 5% : ' + str(round(((one_percent_cnt + exact_cnt + five_percent_cnt) / counter)*100, 1)) + '%')
print('Percent within 10% : ' + str(round(((one_percent_cnt + exact_cnt + five_percent_cnt + ten_percent_cnt) / counter)*100, 1)) + '%')
print('Percent within 20% : ' + str(round(((one_percent_cnt + exact_cnt + five_percent_cnt + ten_percent_cnt + twenty_percent_cnt) / counter)*100, 1)) + '%')
