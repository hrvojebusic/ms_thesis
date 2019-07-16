import argparse
import os
import random
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
import torch.nn.functional as F

import models
import utils.calculate_log as calculate_log
import utils.data_loader as data_loader

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True, help='path to datasets')
parser.add_argument('--data_normalization', action='store_true', help='normalize datasets')
parser.add_argument('--in_dataset', required=True, help='dataset used for training: mnist | cifar10')
parser.add_argument('--out_datasets', required=True, nargs='+', help='datasets used for testing')
parser.add_argument('--out_folder', required=True, help='folder to output results to')
parser.add_argument('--name', type=str, help='name of the run that this test is associated with')
parser.add_argument('--chp_folder', required=True, help='folder with model checkpoints')
parser.add_argument('--chp_prefix', required=True, help='prefix of model checkpoint names')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--cuda', action='store_true', help='use GPU')
parser.add_argument('--gpu_index', type=int, default=0, help='GPU used for training, defaults to: "0"')
args = parser.parse_args()

# Check
print(args)

image_size = 32  # Immutable

if args.cuda:
    gpu_index = args.gpu_index
    print('GPU index: %d' % gpu_index)
else:
    gpu_index = None
    print('No GPU.')

_, in_test_data, channels, classes = data_loader.get_train_data(
    args.in_dataset, args.data_root, args.batch_size, normalize=args.data_normalization
)

out_test_data = []
for dataset in args.out_datasets:
    test_data, _, _ = data_loader.get_test_data(
            dataset, args.data_root, args.batch_size, normalize=args.data_normalization
        )
    out_test_data.append((dataset, test_data))

checkpoints = [
    f for f in listdir(args.chp_folder) if f.startswith(args.chp_prefix) and isfile(join(args.chp_folder, f))
]
checkpoints = [(int(x.split('.')[0].split('_')[-1]), x) for x in checkpoints]
checkpoints.sort(key=lambda x: x[0])

name = args.name if args.name else str(random.randint(0, 2 ** 31 - 1))
f_in_path = os.path.join(args.out_folder, 'peer_review_%s_dataset_%s_report_in.txt' % (name, args.in_dataset))
f_out_path = os.path.join(args.out_folder, 'peer_review_%s_dataset_%s_report_out.txt' % (name, args.in_dataset))

print('Run associated with the test: %s' % name)
print('Training data (in-distribution): %s' % args.in_dataset)
print('Test data (out-distributions): {}'.format(args.out_datasets))
print('Saving in-distribution performance results to:\n%s' % f_in_path)
print('Saving out-distribution performance results to:\n%s' % f_out_path)

with open(f_in_path, 'w') as report_in, open(f_out_path, 'w') as report_out:
    # Headers
    report_in.write('Checkpoint\tTest Accuracy\n')
    report_out.write('Checkpoint\tOut Dataset\tTNR at TPR 95%\tAUROC\tDetection Accuracy\tAUPR In\tAUPR Out\n')

    for chp_no, chp in checkpoints:
        # Model from checkpoint
        model = models.vgg13()  # Added
        #model = models.get_classifier(channels, classes)
        model.load_state_dict(torch.load(join(args.chp_folder, chp)))
        if args.cuda:
            model = model.cuda(gpu_index)
        model.eval()

        #########################
        # In-distribution tests #
        #########################
        tmp_rpt_in = []

        correct = 0
        total = 0

        for data, target in in_test_data:
            total += data.size(0)

            if args.cuda:
                data = data.cuda(gpu_index)
                target = target.cuda(gpu_index)

            batch_output = model(data)
            pred = batch_output.data.max(1)[1]
            equal_flag = pred.eq(target.data).cpu()
            correct += equal_flag.sum()

            for i in range(data.size(0)):  # Confidence score: max_y p(y|x)
                output = batch_output[i].view(1, -1)
                softmax_out = F.softmax(output, dim=1)
                softmax_out = torch.max(softmax_out.data)
                tmp_rpt_in.append(softmax_out.item())

        tmp_rpt_in = np.around(tmp_rpt_in, decimals=5)

        correct, total = int(correct), int(total)
        in_accuracy = 100. * correct / total

        report_in.write('%d\t%.2f\n' % (chp_no, in_accuracy))
        report_in.flush()

        # In metrics
        print('Checkpoint %d' % chp_no)
        print('{:20}{:13.2f}%'.format('Test Set Accuracy:', in_accuracy))

        ##########################
        # Out-distribution tests #
        ##########################
        for dataset, test_loader in out_test_data:
            tmp_rpt_out = []

            for data, _ in test_loader:
                if args.cuda:
                    data = data.cuda(gpu_index)

                batch_output = model(data)
                for i in range(data.size(0)):  # Confidence score: max_y p(y|x)
                    output = batch_output[i].view(1, -1)
                    softmax_out = F.softmax(output, dim=1)
                    softmax_out = torch.max(softmax_out.data)
                    tmp_rpt_out.append(softmax_out.item())

            tmp_rpt_out = np.around(tmp_rpt_out, decimals=5)
            
            tnr_at_tpr_95 = calculate_log.tnr_at_tpr_95(tmp_rpt_in, tmp_rpt_out)
            auroc = calculate_log.auroc(tmp_rpt_in, tmp_rpt_out)
            det_acc = calculate_log.det_acc(tmp_rpt_in, tmp_rpt_out)
            aupr_in = calculate_log.aupr_in(tmp_rpt_in, tmp_rpt_out)
            aupr_out = calculate_log.aupr_out(tmp_rpt_in, tmp_rpt_out)
            
            line = '%d\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
                chp_no, dataset, tnr_at_tpr_95 * 100, auroc * 100, det_acc * 100, aupr_in * 100, aupr_out * 100
            )
            report_out.write(line)
            report_out.flush()

            print('Performance of detector on ' + dataset)
            print('{:20}{:13.3f}%'.format('TNR at TPR 95%:', tnr_at_tpr_95 * 100))
            print('{:20}{:13.3f}%'.format('AUROC:', auroc * 100))
            print('{:20}{:13.3f}%'.format('Detection Accuracy:', det_acc * 100))
            print('{:20}{:13.3f}%'.format('AUPR In:', aupr_in * 100))
            print('{:20}{:13.3f}%'.format('AUPR Out:', aupr_out * 100))
