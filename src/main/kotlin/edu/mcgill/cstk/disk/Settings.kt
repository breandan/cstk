package edu.mcgill.cstk.disk

import ai.djl.modality.nlp.bert.BertTokenizer
import ai.djl.nn.transformer.BertBlock
import org.apache.commons.vfs2.FileExtensionSelector
import java.io.File
import java.net.URL

val FILE_EXTs = setOf("java", "kt")
const val FILE_SCHEME = "file"
const val TGZ_SCHEME = "tgz"
const val HTTP_SCHEME = "http"
val VFS_SELECTOR = FileExtensionSelector(FILE_EXTs)
val VOCAB = object {}::class.java.getResource("/codebert/vocab.json")

val DELIMITER = Regex("\\W")

// https://huggingface.co/microsoft
val MODEL =
// The following models support masking
//"microsoft/codebert-base"
//"microsoft/graphcodebert-base"
"microsoft/codebert-base-mlm"
//"dbernsohn/roberta-java"
//These models do not support masking
//"microsoft/codeGPT-small-java"
//"microsoft/CodeGPT-small-java-adaptedGPT2"
//"microsoft/CodeGPT-small-py"
//"microsoft/CodeGPT-small-py-adaptedGPT2"

val VOCAB_URL = "https://huggingface.co/$MODEL/resolve/main/vocab.json"

val MODEL_DICT: Map<String, Int> by lazy {
  val vocabFile = File("model_$MODEL.json")
  val json = if(vocabFile.exists()) vocabFile.readText()
  else URL(VOCAB_URL).readText().also{vocabFile.writeText(it)}

  json.removePrefix("{\"")
    .substringBeforeLast("\"")
    .split(Regex(", \""))
//    .replace("Ġ", " ") //https://github.com/huggingface/transformers/issues/3867#issuecomment-616956437
    .mapNotNull { it.split("\": ").let { if(it.size == 2) it[0] to it[1].toInt() else null } }
    .toMap()
}

val EMBEDDING_SERVER: String by lazy {
  val addr = "http://localhost:8000/?query="
  val test = addr + "test"

  URL(test).run {
    try {
      if (readText().isNotEmpty()) return@lazy addr
    } catch (ex: Exception) {}
  }

  restartServer()

  println("Starting embeddings server...")
  // Spinlock until service is available
  while (true) try {
    if (URL(test).readText().isNotEmpty()) break
  } catch (exception: Exception) {}

  println("Started embeddings server at $addr")

  addr
}

fun restartServer(): Unit =
  ProcessBuilder("python", "embedding_server.py", "--model=$MODEL", "--offline")
    .also { println("> " + it.command().joinToString(" ")) }
    .run { inheritIO() } // Process will die after a while if this isn't enabled, but it also survives after Ctrl+C
    .start()
    .run { Runtime.getRuntime().addShutdownHook(Thread { println("Server went down!"); destroy(); restartServer() }) }

// Returns the Cartesian product of two sets
operator fun <T, Y> Set<T>.times(s: Set<Y>): Set<Pair<T, Y>> =
  flatMap { l -> s.map { r -> l to r }.toSet() }.toSet()

const val UNK = "<unk>"
const val CLS = "<cls>"
const val SEP = "<sep>"
const val MSK = "<mask>"
const val ERR = "<???>"

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
const val DEFAULT_KNNINDEX_FILENAME = "vector.idx"
const val DEFAULT_KWINDEX_FILENAME = "keyword.idx"
const val MINIGITHUB_REPOS_FILE = "repositories.txt"
const val MINIGITHUB_SIZE = 100
val TOKENIZER = BertTokenizer()