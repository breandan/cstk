package edu.mcgill.gymfs.disk

import ai.djl.modality.nlp.bert.BertTokenizer
import ai.djl.nn.transformer.BertBlock
import org.apache.commons.vfs2.FileExtensionSelector
import java.io.File
import java.net.*
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeUnit.SECONDS
import kotlin.random.Random

const val FILE_EXT = "kt"
const val FILE_SCHEME = "file"
const val TGZ_SCHEME = "tgz"
const val HTTP_SCHEME = "http"
val VFS_SELECTOR = FileExtensionSelector(setOf(FILE_EXT))
val VOCAB = object {}::class.java.getResource("/codebert/vocab.json")

val DELIMITER = Regex("\\W")

val SERVER_ADDRESS by lazy {
  ProcessBuilder("python", "codebert_server.py").start()

  val addr = "http://localhost:8000/?vectorize="
  // Spinlock until service is available
  val startTime = System.currentTimeMillis()

  println("Starting embeddings server...")

  while (true) try {
    if (URL(addr + "test").readText().isNotEmpty()) break
  } catch (exception: Exception) {}

  println("Started embeddings server at $addr")

  addr
}
const val UNK = "<unk>"
const val CLS = "<cls>"
const val SEP = "<sep>"
const val MSK = "<msk>"

const val BERT_EMBEDDING_SIZE = 768
//https://huggingface.co/microsoft/codebert-base/blob/main/special_tokens_map.json
const val CODEBERT_CLS_TOKEN = "<s>"

const val MAX_SEQUENCE_LENGTH = 128
const val MAX_MASKING_PER_INSTANCE = 20
const val BATCH_SIZE = 24
const val MAX_BATCH = 50
const val MAX_GPUS = 1
const val EPOCHS = 100000
const val MAX_VOCAB = 35000

val RESERVED_TOKENS = listOf(UNK, CLS, SEP, MSK)
val UNK_ID = RESERVED_TOKENS.indexOf(UNK)
val CLS_ID = RESERVED_TOKENS.indexOf(CLS)
val SEP_ID = RESERVED_TOKENS.indexOf(SEP)
val MSK_ID = RESERVED_TOKENS.indexOf(MSK)
val BERT_BUILDER = BertBlock.builder().micro()
val ROOT_DIR = File("").absoluteFile.toURI()
val DATA_DIR = File("data").absoluteFile.toURI()
val TEST_DIR = File("src").absoluteFile.toURI()
val DEFAULT_KNNINDEX_FILENAME = "vector.idx"
val DEFAULT_KWINDEX_FILENAME = "keyword.idx"
val MINIGITHUB_REPOS_FILE = "repositories.txt"
val MINIGITHUB_SIZE = 100
val TOKENIZER = BertTokenizer()
val rand = Random(1)