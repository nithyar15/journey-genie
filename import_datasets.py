import argparse
import os
import shutil
from os.path import exists

from assistant_utils import process_assistant

from nemo.collections.nlp.data.data_utils.data_preprocessing import (
    DATABASE_EXISTS_TMP,
    MODE_EXISTS_TMP,
    create_dataset,
    get_dataset,
    get_vocab,
    if_exist,
)
from nemo.utils import logging


def ids2text(ids, vocab):
    return ' '.join([vocab[int(id_)] for id_ in ids])


def process_journeygenie(infold, outfold, modes=['train', 'test'], do_lower_case=False):
    vocab = get_vocab(f'{infold}/journeygenie.dict.vocab.csv')

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logging.info(DATABASE_EXISTS_TMP.format('journeygenie', outfold))
        return outfold
    logging.info(f'Processing journeygenie dataset and storing at {outfold}.')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}
    for mode in modes:
        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w', encoding='utf-8')
        outfiles[mode].write('sentence\tlabel\n')
        outfiles[mode + '_slots'] = open(f'{outfold}/{mode}_slots.tsv', 'w', encoding='utf-8')

        queries = open(f'{infold}/journeygenie.{mode}.query.csv', 'r', encoding='utf-8').readlines()
        intents = open(f'{infold}/journeygenie.{mode}.intent.csv', 'r', encoding='utf-8').readlines()
        slots = open(f'{infold}/journeygenie.{mode}.slots.csv', 'r', encoding='utf-8').readlines()

        for i, query in enumerate(queries):
            sentence = ids2text(query.strip().split()[1:-1], vocab)
            if do_lower_case:
                sentence = sentence.lower()
            outfiles[mode].write(f'{sentence}\t{intents[i].strip()}\n')
            slot = ' '.join(slots[i].strip().split()[1:-1])
            outfiles[mode + '_slots'].write(slot + '\n')

    shutil.copyfile(f'{infold}/journeygenie.dict.intent.csv', f'{outfold}/dict.intents.csv')
    shutil.copyfile(f'{infold}/journeygenie.dict.slots.csv', f'{outfold}/dict.slots.csv')
    for mode in modes:
        outfiles[mode].close()
