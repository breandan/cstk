package edu.mcgill.cstk.disk

import ai.djl.modality.nlp.bert.BertTokenizer
import ai.djl.nn.transformer.BertBlock
import edu.mcgill.cstk.experiments.probing.embeddingServer
import org.apache.commons.vfs2.FileExtensionSelector
import java.io.File
import java.net.URL
import kotlin.random.Random

val FILE_EXTs = setOf("java", "kt")
const val FILE_SCHEME = "file"
const val TGZ_SCHEME = "tgz"
const val ZIP_SCHEME = "zip"
const val HTTP_SCHEME = "http"
val VFS_SELECTOR = FileExtensionSelector(FILE_EXTs)
val VOCAB = object {}::class.java.getResource("/codebert/vocab.json")

val DELIMITER = Regex("\\W")

// https://huggingface.co/microsoft
val MODELS = setOf(
  "microsoft/codebert-base-mlm",
  "huggingface/CodeBERTa-small-v1",
  "microsoft/graphcodebert-base",
  "dbernsohn/roberta-java",
).map { Model(it) }.toSet()

data class Model(val name: String) {
  val mask = "<mask>"
  override fun hashCode() = name.hashCode()
  override fun toString() = name
  override fun equals(other: Any?) = (other as? Model)?.name == name
}

// The following models support masking
//"microsoft/graphcodebert-base"
//"microsoft/codebert-base-mlm"
//"dbernsohn/roberta-java"
//These models do not support masking
//"microsoft/codebert-base"
//"microsoft/codeGPT-small-java"
//"microsoft/CodeGPT-small-java-adaptedGPT2"
//"microsoft/CodeGPT-small-py"
//"microsoft/CodeGPT-small-py-adaptedGPT2"

//val VOCAB_URL = "https://huggingface.co/$MODEL/resolve/main/vocab.json"
//val MODEL_DICT: Map<String, Int> by lazy {
//  val vocabFile = File("model_${MODEL.replace("/", "_")}.json")
//  val json = if (vocabFile.exists()) vocabFile.readText()
//  else URL(VOCAB_URL).readText().also { vocabFile.run { createNewFile(); writeText(it) } }
//
//  json.removePrefix("{\"")
//    .substringBeforeLast("\"")
//    .split(Regex(", \""))
////    .replace("Ġ", " ") //https://github.com/huggingface/transformers/issues/3867#issuecomment-616956437
//    .mapNotNull { it.split("\": ").let { if (it.size == 2) it[0] to it[1].toInt() else null } }
//    .toMap()
//}

val defaultModel = MODELS.first()

val SERVER_URL: String by lazy {
  val addr = "http://localhost:8000/"
  val url = URL("$addr${defaultModel.name}?query=test")

  println("Default URL: $url")
  if (url.runCatching { readText() }.isSuccess) return@lazy addr

  restartServer()

  println("Starting embeddings server...")
  // Spinlock until service is available
  while (true) {
    Thread.sleep(1000)
    if (url.runCatching { readText() }.getOrDefault("").isNotEmpty())
      break
  }

  println("Started embeddings server at $addr")

  addr
}

fun shouldBeOffline(): String =
  try {
    if (URL("https://huggingface.co/").readText().isNotEmpty()) {
      println("Connected to HuggingFace successfully.")
    }
    System.getenv("TRANSFORMERS_OFFLINE")
      .takeUnless { it.isNullOrEmpty() }
      ?.let {
        if (it == "1")
          "--offline".also { println("Using $it due to environment variable") }
        else ""
      } ?: ""
  } catch (exception: Exception) {
    "--offline"
  }

fun restartServer(): Unit =
  try {
    val models = MODELS.map { it.name }.toTypedArray()
    val offline = "--offline"//shouldBeOffline()

    ProcessBuilder(
//      "bash", "-c",
//      "source", "venv/bin/activate", "&&",
//      "while", "true;", "do",
      "python", embeddingServer.absolutePath, "--models", *models, offline
//      "&&", "break;", "done"
    ).also { println("> " + it.command().joinToString(" ")) }
     .run { inheritIO() } // Process will die after a while if this isn't enabled, but it also survives after Ctrl+C
     .start().run {
       Runtime.getRuntime().addShutdownHook(Thread {
         println("Server went down!")
         destroy()
//         restartServer()
       })
     }
  } catch (ex: Exception) {
    ex.printStackTrace()
//    restartServer()
  }

/** Defaults, configure custom special tokens in [Model] */
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

val RESERVED_TOKENS = listOf(UNK, CLS, SEP, MSK, "[MASK]")
val UNK_ID = RESERVED_TOKENS.indexOf(UNK)
val BERT_BUILDER = BertBlock.builder().micro()
val ROOT_DIR = File("").absoluteFile.toURI()
val DATA_DIR = File("data").absoluteFile.toURI()
val GCODE_DIR = File("data/gcode").absoluteFile.toURI()
val CSN_DIR = File("data/CSN").absoluteFile.toURI()
val TEST_DIR = File("src").absoluteFile.toURI()
const val DEFAULT_KNNINDEX_FILENAME = "vector.idx"
const val DEFAULT_KWINDEX_FILENAME = "keyword.idx"
val TOKENIZER = BertTokenizer()

val DEFAULT_RAND = Random(1)