package edu.mcgill.gymfs.disk

import ai.djl.modality.nlp.bert.BertTokenizer
import ai.djl.nn.transformer.BertBlock
import org.apache.commons.vfs2.FileExtensionSelector
import java.io.File
import java.net.URL

const val FILE_EXT = "kt"
const val FILE_SCHEME = "file"
const val TGZ_SCHEME = "tgz"
const val HTTP_SCHEME = "http"
val VFS_SELECTOR = FileExtensionSelector(setOf(FILE_EXT))
val VOCAB = object {}::class.java.getResource("/codebert/vocab.json")

val DELIMITER = Regex("\\W")

// https://huggingface.co/microsoft
val MODEL =
//"codeGPT-small-java"
//"CodeGPT-small-java-adaptedGPT2"
//"CodeGPT-small-py"
//"CodeGPT-small-py-adaptedGPT2"
"graphcodebert-base"
//"codebert-base"
//"codebert-base-mlm"

val EMBEDDING_SERVER by lazy {
  val addr = "http://localhost:8000/?query="

  if (URL(addr + "test").readText().isNotEmpty()) return@lazy addr

  ProcessBuilder("python", "embedding_server.py", MODEL)
//    .run { inheritIO() }
    .start()

  println("Starting embeddings server...")
  // Spinlock until service is available
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