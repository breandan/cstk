package edu.mcgill.gymfs

import ai.djl.modality.nlp.bert.BertTokenizer
import ai.djl.nn.transformer.BertBlock
import java.io.File
import kotlin.random.Random
import kotlin.system.exitProcess

const val FILE_EXT = "*.kt"

const val UNK = "<unk>"
const val CLS = "<cls>"
const val SEP = "<sep>"
const val MSK = "<msk>"

const val MAX_SEQUENCE_LENGTH = 128
const val MAX_MASKING_PER_INSTANCE = 20
const val BATCH_SIZE = 24
const val MAX_BATCH = 50
const val MAX_GPUS = 1
const val EPOCHS = 1000
const val MAX_VOCAB = 35000

val RESERVED_TOKENS = listOf(UNK, CLS, SEP, MSK)
val UNK_ID = RESERVED_TOKENS.indexOf(UNK)
val CLS_ID = RESERVED_TOKENS.indexOf(CLS)
val SEP_ID = RESERVED_TOKENS.indexOf(SEP)
val MSK_ID = RESERVED_TOKENS.indexOf(MSK)
val BERT_BUILDER = BertBlock.builder().micro()
val ROOT_DIR = File(".").toPath()
val TOKENIZER = BertTokenizer()
val rand = Random(1)