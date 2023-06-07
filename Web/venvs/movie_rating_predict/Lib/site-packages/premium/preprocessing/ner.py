#!/usr/bin/env python3
from typing import List

import pandas as pd
from codefast.ds import flatten


def generate_label_maps(labels: List[str], seperator=','):
    """ Generate label maps from labels
    Args:
        labels: list of labels
        seperator: seperator of labels
    Returns:
        tag2id: tag to id mapping
        id2tag: id to tag mapping
    """
    gen = flatten([label.split(seperator) for label in labels])
    entities = set([x for x in gen])
    tag2id = {tag: idx for idx, tag in enumerate(entities)}
    id2tag = {idx: tag for idx, tag in tag2id.items()}
    return tag2id, id2tag


def format_to_csv(source_file: str, target_file: str, reverse_text_label=False):
    """read file from source_file, reformat it and save to target_file
    The format of source file is as follows:
    ```
    O	that
    O	movie
    B-Plot	aliens
    I-Plot	invading
    I-Plot	earth
    I-Plot	in
    I-Plot	a
    I-Plot	particular
    I-Plot	united
    I-Plot	states
    I-Plot	place
    I-Plot	in
    I-Plot	california

    O	what
    B-Genre	soviet
    I-Genre	science
    I-Genre	fiction
    B-Opinion	classic
    ```
    """
    df = pd.DataFrame(columns=['text', 'label'])
    text, label,data = [], [],[]
    with open(source_file, 'r') as f:
        for line in f:
            if line.strip():
                t, l = line.strip().split('\t')
                if reverse_text_label:
                    t, l = l, t
                text.append(t)
                label.append(l)
            else:
                data.append((text, label))
                text, label = [], []
    df['text'] = [' '.join(text) for text, _ in data]
    df['label'] = [','.join(label) for _, label in data]
    df.to_csv(target_file, index=False)


def extract_bio(text: str) -> List[str]:
    """ Get NER BIO sequence from text
    """
    text = text.strip()
    if not text:
        return []
    bio = []
    amid = False
    for c in text:
        if c == '[':
            amid = True
        elif c == ']':
            amid = False
        else:
            if amid:
                if (bio and bio[-1] == 'O') or not bio:
                    bio.append('B')
                else:
                    bio.append('I')
            else:
                bio.append('O')
    return bio


def extract_bioe(text: str) -> List[str]:
    """ Get NER BIOE sequence from text
    """
    text = text.strip()
    bioe, stack = [], []
    i = 0
    while i < len(text):
        c = text[i]
        if c == '[':
            i += 1
            while i < len(text) and text[i] != ']':
                stack.append(c)
                i += 1
            if len(stack) == 1:
                bioe.append('B')
            elif len(stack) == 2:
                bioe.extend(['B', 'E'])
            else:
                bioe.extend(['B'] + ['I'] * (len(stack) - 2) + ['E'])
        else:
            bioe.append('O')
        i += 1

    return bioe
