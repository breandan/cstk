package edu.mcgill.gymfs

import ai.djl.modality.nlp.bert.BertTokenizer
import ai.djl.nn.transformer.BertBlock
import kotlin.random.Random

val UNK = "<unk>"
val CLS = "<cls>"
val SEP = "<sep>"
val MSK = "<msk>"
val RESERVED_TOKENS = listOf(UNK, CLS, SEP, MSK)
val UNK_ID = RESERVED_TOKENS.indexOf(UNK)
val CLS_ID = RESERVED_TOKENS.indexOf(CLS)
val SEP_ID = RESERVED_TOKENS.indexOf(SEP)
val MSK_ID = RESERVED_TOKENS.indexOf(MSK)
val MAX_SEQUENCE_LENGTH = 128
val MAX_MASKING_PER_INSTANCE = 20
val BATCH_SIZE = 24
val MAX_BATCH = 50
val MAX_GPUS = 1
val EPOCHS = 10
val BERT_BUILDER = BertBlock.builder().micro()

val MAX_VOCAB = 35000
val TOKENIZER = BertTokenizer()
val rand = Random(1)