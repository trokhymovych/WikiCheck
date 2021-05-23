import csv
import sys


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def _read_csv(input_file, quotechar=None):
    """Reads a comma separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=",", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def _create_examples_snli(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, line[0])
        text_a = line[7]
        text_b = line[8]
        label = line[-1]
        examples.append([guid, text_a, text_b, label])
    return examples


def _create_examples_mnli(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for i, line in lines.iterrows():
        if i == 0:
            continue
        text_a = str(line.sentence1)
        text_b = str(line.sentence2)
        label = line.gold_label
        examples.append([i, text_a, text_b, label])
    return examples


def _create_examples_fever(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for i, line in lines.iterrows():
        if i == 0:
            continue
        text_a = line.claim
        text_b = line.hypothesis
        label = line.label
        examples.append([i, text_a, text_b, label])
    return examples