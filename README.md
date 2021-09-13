# CSTK: Code Search Toolkit

Tools and experiments for code search. Provides:

* Indices for keyword and vector embedding
* Learning to search & grammar induction
    * Passive DFA learning from membership
    * Keyword/BoW-based query synthesis
* Semantic graph construction
    * Keyword-matching edge construction
    * Proximity-based graph embedding
* Metrics for string, vector and distribution matching
    * Kantorovich metric on code embeddings
    * Code-snippet normal form
    * Various string distance metrics
* TSNE visualization of code embeddings
* Ranking metrics: NDCG, MAP@K, MRR
* Datamining and dataloading tools
* ["MiniGitHub" mock training interface](#minigithub-construction)
* [Probabilistic code synthesis with Markov tensors](#probabilistic-code-synthesis)
* Probing tools for pretrained neural language models
* Autoregressive code completion with masked LMs
* Synthetic source code transformations
    * [Synonym variable renaming](#synonym-renaming)
    * Dead code introduction
    * Loop bounds alteration
    * Argument order swapping
    * Line order swapping
* [Method slicing](latex/notes/slicing.pdf)

# Indexing

Provides a mock API for filesystem interactions.

Stores BPE-compressed files in memory.

Gives an agent the ability to selectively read and query files.

Interface:

* `Path.read(start, end)` - Returns file chunk at offset.
* `Path.grep(query)` - Returns offsets matching query.
* `Path.knn(code)` - Fetches similar code snippets to the query.

# Usage

### MiniGitHub construction

Clones a bunch of smallish repos on GitHub for evaluation:

```bash
./gradlew cloneRepos
```

Downloads Git repos into the `data` directory by default.

### Keyword search

How quickly can we search for substrings? Useful for learning to search.

```bash
./gradlew -q trieSearch --args='--query=<QUERY> [--path=<PATH_TO_INDEX>] [--index=<INDEX_FILE>]'
```

<details>

```
$ ./gradlew -q trieSearch
Indexing /home/breandan/IdeaProjects/gym-fs
Indexed in 524ms to: gymfs.idx

Searching index of size 1227 for [?]=[match]…

0.) [?=match] ….default("[?]")… (…Environment.kt:L21)
Keyword scores: [(toAbsolutePath, 2.0), (Query, 2.0), (find, 2.0)]
Next locations:
        0.) [?=toAbsolutePath] …ath = src.[?]().toStrin…        (…DiskUtils.kt:L21)
        1.) [?=toAbsolutePath] …s.get("").[?]().toStrin…        (…Environment.kt:L19)
        2.) [?=Query] …// [?] in contex…        (…StringUtils.kt:L7)
        3.) [?=find] …ex(query).[?]All(this).…  (…StringUtils.kt:L36)

1.) [?=match] …val ([?]Start, mat…tchStart, [?]End) =…  (…StringUtils.kt:L38)
Keyword scores: [(Regex, 2.0), (matchStart, 2.0), (matchEnd, 2.0)]
Next locations:
        0.) [?=Regex] …(3).split([?]("[^\\w']+… (…Environment.kt:L66)
        1.) [?=Regex] …[?](query).fi…   (…StringUtils.kt:L36)
        2.) [?=matchStart] …substring([?], matchEnd…chEnd) to [?]…      (…StringUtils.kt:L40)
        3.) [?=matchEnd] …tchStart, [?]) to match…      (…StringUtils.kt:L40)

2.) [?=match] …substring([?]Start, mat…tchStart, [?]End) to ma…chEnd) to [?]Start…      (…StringUtils.kt:L40)
Keyword scores: [(matchStart, 2.0), (matchEnd, 2.0), (first, 3.0)]
Next locations:
        0.) [?=matchStart] …val ([?], matchEnd… (…StringUtils.kt:L38)
        1.) [?=matchEnd] …tchStart, [?]) =…     (…StringUtils.kt:L38)
        2.) [?=first] ….offer(it.[?]()) }…      (…Environment.kt:L120)
        3.) [?=first] …st common [?]. Common k… (…Environment.kt:L77)
        4.) [?=first] …it.range.[?].coerceIn(…  (…StringUtils.kt:L39)

3.) [?=match] …pairs of [?]ing prefix…  (…Environment.kt:L25)
Keyword scores: [(offset, 2.0), (pairs, 2.0), (help, 3.0)]
Next locations:
        0.) [?=offset] …val [?]: Int…   (…StringUtils.kt:L12)
        1.) [?=pairs] …sentence [?] containin…  (…BertTrainer.kt:L112)
        2.) [?=help] …--index", [?] = "Prebui…  (…Environment.kt:L23)
        3.) [?=help] …--query", [?] = "Query…   (…Environment.kt:L21)
        4.) [?=help] …"--path", [?] = "Root d…  (…Environment.kt:L18)


Found 4 results in 2.82ms
```

</details>

### Nearest neighbor search

What do k-nearest neighbors look like?

```bash
./gradlew -q knnSearch --args='--query=<QUERY> [--path=<PATH_TO_INDEX>] [--index=<INDEX_FILE>] [--graphs=10]'
```

<details>

```
$ ./gradlew -q knnSearch --args='--query="const val MAX_GPUS = 1"'

Searching KNN index of size 981 for [?]=[const val MAX_GPUS = 1]…

0.) const val MAX_GPUS = 1
1.) const val MAX_BATCH = 50
2.) const val MAX_VOCAB = 35000
3.) const val EPOCHS = 100000
4.) const val BATCH_SIZE = 24
5.) const val MAX_SEQUENCE_LENGTH = 128
6.) const val CLS = "<cls>"
7.) dataSize.toLong()
8.) const val BERT_EMBEDDING_SIZE = 768
9.) const val UNK = "<unk>"

Fetched nearest neighbors in 1.48674ms

|-----> Original index before reranking by MetricLCS
|    |-----> Current index after reranking by MetricLCS
|    |
  0->0.) const val MAX_GPUS = 1
  1->1.) const val MAX_BATCH = 50
 14->2.) const val MSK = "<msk>"
  3->3.) const val EPOCHS = 100000
  4->4.) const val BATCH_SIZE = 24
  2->5.) const val MAX_VOCAB = 35000
363->6.) ).default("const val MAX_GPUS = 1")
  6->7.) const val CLS = "<cls>"
  9->8.) const val UNK = "<unk>"
 16->9.) const val SEP = "<sep>"

Reranked nearest neighbors in 1.412775ms
```
</details>

### Semantic vs. Syntactic Similarity

What do nearest neighbors share in common?

```bash
./gradlew nearestNeighbors
```

<details>

```
$ ./gradlew nearestNeighbors

Angle brackets enclose longest common substring up to current result

0.] dataSize.toLong()
	0.0] executorService.shutdownNow()
	0.1] PolynomialDecayTracker.builder《()》
	0.2] .toLabeledGraph《()》
	0.3] WarmUpTracker.builder《()》
	0.4] .allCodeFragments《()》
	0.5] .toTypedArray《()》
	0.6] batchData: TrainingListener.BatchData
	0.7] .asSequence()
	0.8] .shuffled()
	0.9] .readText().lines()
	0.10] vocabSize
	0.11] .toList()
	0.12] .distinct()
	0.13] PaddingStackBatchifier.builder()
	0.14] return trainer.trainingResult
	0.15] Adam.builder()
	0.16] return jfsRoot
	0.17] createOrLoadModel()
	0.18] sentenceA = otherA
	0.19] const val MAX_GPUS = 1


1.] .toLabeledGraph()
	1.0] .toTypedArray()
	1.1] 《.to》List()
	1.2] .asSequence《()》
	1.3] .allCodeFragments《()》
	1.4] .renderVKG《()》
	1.5] .shuffled《()》
	1.6] .distinct《()》
	1.7] dataSize.toLong《()》
	1.8] .readText《()》.lines《()》
	1.9] PolynomialDecayTracker.builder《()》
	1.10] WarmUpTracker.builder《()》
	1.11] .show《()》
	1.12] .readText《()》
	1.13] Adam.builder《()》
	1.14] .allFilesRecursively《()》
	1.15] executorService.shutdownNow《()》
	1.16] .build《()》
	1.17] .first《()》.toDoubleArray《()》
	1.18] PaddingStackBatchifier.builder《()》
	1.19] .optLimit(100)


2.] .shuffled()
	2.0] .distinct()
	2.1] .renderVKG《()》
	2.2] .toLabeledGraph《()》
	2.3] .show《()》
	2.4] .toTypedArray《()》
	2.5] .toList《()》
	2.6] .asSequence《()》
	2.7] .allCodeFragments《()》
	2.8] .build《()》
	2.9] dataSize.toLong《()》
	2.10] .readText《()》.lines《()》
	2.11] PolynomialDecayTracker.builder《()》
	2.12] WarmUpTracker.builder《()》
	2.13] .allFilesRecursively《()》
	2.14] .first《()》.toDoubleArray《()》
	2.15] executorService.shutdownNow《()》
	2.16] .readText《()》
	2.17] PaddingStackBatchifier.builder《()》
	2.18] trainer.metrics = Metrics《()》
	2.19] Adam.builder《()》


3.] .toList()
	3.0] .toTypedArray()
	3.1] 《.to》LabeledGraph()
	3.2] .distinct《()》
	3.3] .asSequence《()》
	3.4] .shuffled《()》
	3.5] .readText《()》.lines《()》
	3.6] .allCodeFragments《()》
	3.7] .show《()》
	3.8] .allFilesRecursively《()》
	3.9] dataSize.toLong《()》
	3.10] .renderVKG《()》
	3.11] .readText《()》
	3.12] .build《()》
	3.13] WarmUpTracker.builder《()》
	3.14] .first《()》.toDoubleArray《()》
	3.15] PolynomialDecayTracker.builder《()》
	3.16] executorService.shutdownNow《()》
	3.17] trainer.metrics = Metrics《()》
	3.18] Adam.builder《()》
	3.19] .optLimit(100)


4.] PolynomialDecayTracker.builder()
	4.0] WarmUpTracker.builder()
	4.1] PaddingStackBatchifi《er.builder()》
	4.2] dataSize.toLong《()》
	4.3] TrainBertOnCode.runExample《()》
	4.4] executorService.shutdownNow《()》
	4.5] trainer.metrics = Metrics《()》
	4.6] .shuffled《()》
	4.7] .toLabeledGraph《()》
	4.8] .toTypedArray《()》
	4.9] .distinct《()》
	4.10] createOrLoadModel《()》
	4.11] Activation.relu(it)
	4.12] .renderVKG()
	4.13] batchData: TrainingListener.BatchData
	4.14] else rebuildIndex()
	4.15] .allCodeFragments()
	4.16] return jfsRoot
	4.17] .asSequence()
	4.18] .toList()
	4.19] vocabSize


5.] .distinct()
	5.0] .shuffled()
	5.1] 《.sh》ow()
	5.2] .toList《()》
	5.3] .toLabeledGraph《()》
	5.4] .renderVKG《()》
	5.5] .build《()》
	5.6] .asSequence《()》
	5.7] .toTypedArray《()》
	5.8] dataSize.toLong《()》
	5.9] .readText《()》.lines《()》
	5.10] .allCodeFragments《()》
	5.11] PolynomialDecayTracker.builder《()》
	5.12] WarmUpTracker.builder《()》
	5.13] Adam.builder《()》
	5.14] .allFilesRecursively《()》
	5.15] .readText《()》
	5.16] executorService.shutdownNow《()》
	5.17] trainer.metrics = Metrics《()》
	5.18] createOrLoadModel《()》
	5.19] printQuery《()》


6.] WarmUpTracker.builder()
	6.0] PolynomialDecayTracker.builder()
	6.1] PaddingStackBatchifi《er.builder()》
	6.2] TrainBertOnCode.runExample《()》
	6.3] dataSize.toLong《()》
	6.4] trainer.metrics = Metrics《()》
	6.5] executorService.shutdownNow《()》
	6.6] .shuffled《()》
	6.7] .toTypedArray《()》
	6.8] .distinct《()》
	6.9] .toLabeledGraph《()》
	6.10] Activation.relu(it)
	6.11] .toList()
	6.12] .renderVKG()
	6.13] else rebuildIndex()
	6.14] .asSequence()
	6.15] createOrLoadModel()
	6.16] batchData: TrainingListener.BatchData
	6.17] .allCodeFragments()
	6.18] .readText().lines()
	6.19] TextTerminator()


7.] .toTypedArray()
	7.0] .toLabeledGraph()
	7.1] 《.toL》ist()
	7.2] .asSequence《()》
	7.3] .shuffled《()》
	7.4] .allCodeFragments《()》
	7.5] dataSize.toLong《()》
	7.6] .distinct《()》
	7.7] .renderVKG《()》
	7.8] WarmUpTracker.builder《()》
	7.9] PolynomialDecayTracker.builder《()》
	7.10] .readText《()》.lines《()》
	7.11] .allFilesRecursively《()》
	7.12] .first《()》.toDoubleArray《()》
	7.13] .readText《()》
	7.14] executorService.shutdownNow《()》
	7.15] .show《()》
	7.16] PaddingStackBatchifier.builder《()》
	7.17] trainer.metrics = Metrics《()》
	7.18] .build《()》
	7.19] TrainBertOnCode.runExample《()》


8.] const val MAX_BATCH = 50
	8.0] const val MAX_VOCAB = 35000
	8.1] 《const val MAX_》GPUS = 1
	8.2] 《const val 》EPOCHS = 100000
	8.3] 《const val 》MAX_SEQUENCE_LENGTH = 128
	8.4] 《const val 》BATCH_SIZE = 24
	8.5] 《const val 》CLS = "<cls>"
	8.6] 《const val 》UNK = "<unk>"
	8.7] 《const val 》BERT_EMBEDDING_SIZE = 768
	8.8] dataSize.toL《on》g()
	8.9] val targetEmbedding =
	8.10] const val MSK = "<msk>"
	8.11] val use = UniversalSentenceEncoder
	8.12] sentenceA = otherA
	8.13] const val CODEBERT_CLS_TOKEN = "<s>"
	8.14] const val SEP = "<sep>"
	8.15] val d2vecs = vectors.reduceDim()
	8.16] return jfsRoot
	8.17] val range = 0..length
	8.18] val (matchStart, matchEnd) =
	8.19] PolynomialDecayTracker.builder()


9.] .renderVKG()
	9.0] .toLabeledGraph()
	9.1] .shuff《led》()
	9.2] .allCodeFragments《()》
	9.3] .distinct《()》
	9.4] .show《()》
	9.5] .toTypedArray《()》
	9.6] .toList《()》
	9.7] .readText《()》.lines《()》
	9.8] .build《()》
	9.9] dataSize.toLong《()》
	9.10] PolynomialDecayTracker.builder《()》
	9.11] WarmUpTracker.builder《()》
	9.12] .readText《()》
	9.13] .asSequence《()》
	9.14] .allFilesRecursively《()》
	9.15] Adam.builder《()》
	9.16] printQuery《()》
	9.17] createOrLoadModel《()》
	9.18] TrainBertOnCode.runExample《()》
	9.19] PaddingStackBatchifier.builder《()》


10.] .readText().lines()
	10.0] .readText()
	10.1] .toLabeledGraph《()》
	10.2] .toList《()》
	10.3] dataSize.toLong《()》
	10.4] .shuffled《()》
	10.5] .allCodeFragments《()》
	10.6] path.readText《()》.lines《()》
	10.7] .distinct《()》
	10.8] .renderVKG《()》
	10.9] .toTypedArray《()》
	10.10] .allFilesRecursively《()》
	10.11] .asSequence《()》
	10.12] .show《()》
	10.13] executorService.shutdownNow《()》
	10.14] WarmUpTracker.builder《()》
	10.15] Adam.builder《()》
	10.16] .build《()》
	10.17] PolynomialDecayTracker.builder《()》
	10.18] .first《()》.toDoubleArray《()》
	10.19] trainer.metrics = Metrics《()》


11.] .show()
	11.0] .build()
	11.1] .distinct《()》
	11.2] .shuffled《()》
	11.3] .toList《()》
	11.4] Adam.builder《()》
	11.5] .renderVKG《()》
	11.6] printQuery《()》
	11.7] .toLabeledGraph《()》
	11.8] TextTerminator《()》
	11.9] .readText《()》
	11.10] println《()》
	11.11] createOrLoadModel《()》
	11.12] .readText《()》.lines《()》
	11.13] else rebuildIndex《()》
	11.14] .toTypedArray《()》
	11.15] WarmUpTracker.builder《()》
	11.16] dataSize.toLong《()》
	11.17] .allCodeFragments《()》
	11.18] PolynomialDecayTracker.builder《()》
	11.19] }.toList《()》


12.] const val MAX_VOCAB = 35000
	12.0] const val MAX_BATCH = 50
	12.1] 《const val 》EPOCHS = 100000
	12.2] 《const val 》MAX_GPUS = 1
	12.3] 《const val 》MAX_SEQUENCE_LENGTH = 128
	12.4] 《const val 》CLS = "<cls>"
	12.5] 《const val 》BATCH_SIZE = 24
	12.6] 《const val 》UNK = "<unk>"
	12.7] 《const val 》MSK = "<msk>"
	12.8] 《const val 》BERT_EMBEDDING_SIZE = 768
	12.9] dataSize.toL《on》g()
	12.10] val d2vecs = vectors.reduceDim()
	12.11] const val SEP = "<sep>"
	12.12] val vocab = SimpleVocabulary.builder()
	12.13] val use = UniversalSentenceEncoder
	12.14] const val CODEBERT_CLS_TOKEN = "<s>"
	12.15] val targetEmbedding =
	12.16] sentenceA = otherA
	12.17] return jfsRoot
	12.18] val r = rand.nextFloat()
	12.19] PolynomialDecayTracker.builder()


13.] .allCodeFragments()
	13.0] .toLabeledGraph()
	13.1] .renderVKG《()》
	13.2] .toTypedArray《()》
	13.3] .allFilesRecursively《()》
	13.4] .toList《()》
	13.5] .shuffled《()》
	13.6] dataSize.toLong《()》
	13.7] .readText《()》.lines《()》
	13.8] .asSequence《()》
	13.9] .distinct《()》
	13.10] PolynomialDecayTracker.builder《()》
	13.11] .readText《()》
	13.12] WarmUpTracker.builder《()》
	13.13] executorService.shutdownNow《()》
	13.14] Adam.builder《()》
	13.15] .show《()》
	13.16] .build《()》
	13.17] .optLimit(100)
	13.18] .optBatchFirst(true)
	13.19] PaddingStackBatchifier.builder()


14.] const val MAX_GPUS = 1
	14.0] const val MAX_BATCH = 50
	14.1] 《const val MAX_》VOCAB = 35000
	14.2] 《const val 》EPOCHS = 100000
	14.3] 《const val 》BATCH_SIZE = 24
	14.4] 《const val 》MAX_SEQUENCE_LENGTH = 128
	14.5] 《const val 》CLS = "<cls>"
	14.6] dataSize.toL《on》g()
	14.7] c《on》st val BERT_EMBEDDING_SIZE = 768
	14.8] c《on》st val UNK = "<unk>"
	14.9] c《on》st val CODEBERT_CLS_TOKEN = "<s>"
	14.10] val targetEmbedding =
	14.11] val use = UniversalSentenceEncoder
	14.12] sentenceA = otherA
	14.13] const val MSK = "<msk>"
	14.14] val (matchStart, matchEnd) =
	14.15] const val SEP = "<sep>"
	14.16] return jfsRoot
	14.17] return trainer.trainingResult
	14.18] var numEpochs = 0
	14.19] PolynomialDecayTracker.builder()


15.] createOrLoadModel()
	15.0] printQuery()
	15.1] TextTerminator《()》
	15.2] else rebuildIndex《()》
	15.3] println《()》
	15.4] TrainBertOnCode.runExample《()》
	15.5] dataSize.toLong《()》
	15.6] PolynomialDecayTracker.builder《()》
	15.7] executorService.shutdownNow《()》
	15.8] return trainer.trainingResult
	15.9] WarmUpTracker.builder()
	15.10] }.toList()
	15.11] .show()
	15.12] PaddingStackBatchifier.builder()
	15.13] add(CLS)
	15.14] .build()
	15.15] Adam.builder()
	15.16] vocabSize
	15.17] .distinct()
	15.18] sentenceA = otherA
	15.19] .shuffled()


16.] vocabSize
	16.0] return trainer.trainingResult
	16.1] 《return 》jfsRoot
	16.2] 《return 》dataset
	16.3] Adam.builder()
	16.4] dataSize.toLong()
	16.5] rootDir: Path
	16.6] sentenceA = otherA
	16.7] val offset: Int
	16.8] list: NDList
	16.9] batchData: TrainingListener.BatchData
	16.10] TextTerminator()
	16.11] executorService.shutdownNow()
	16.12] PolynomialDecayTracker.builder()
	16.13] vocabSize: Long
	16.14] createOrLoadModel()
	16.15] PunctuationSeparator(),
	16.16] TextTruncator(10)
	16.17] Batchifier.STACK,
	16.18] add(CLS)
	16.19] PaddingStackBatchifier.builder()


17.] const val EPOCHS = 100000
	17.0] const val MAX_VOCAB = 35000
	17.1] 《const val MAX_》BATCH = 50
	17.2] 《const val MAX_》GPUS = 1
	17.3] 《const val MAX_》SEQUENCE_LENGTH = 128
	17.4] 《const val 》CLS = "<cls>"
	17.5] 《const val 》BATCH_SIZE = 24
	17.6] 《const val 》UNK = "<unk>"
	17.7] 《const val 》MSK = "<msk>"
	17.8] 《const val 》SEP = "<sep>"
	17.9] 《const val 》BERT_EMBEDDING_SIZE = 768
	17.10] 《val 》targetEmbedding =
	17.11] dataSize.toLong()
	17.12] val use = UniversalSentenceEncoder
	17.13] const val CODEBERT_CLS_TOKEN = "<s>"
	17.14] val d2vecs = vectors.reduceDim()
	17.15] val vocab = SimpleVocabulary.builder()
	17.16] var consecutive = true
	17.17] val knn = knnIndex.findNearest(v, topK)
	17.18] sentenceA = otherA
	17.19] val r = rand.nextFloat()


18.] Adam.builder()
	18.0] return dataset
	18.1] .show()
	18.2] vocabSize
	18.3] TextTerminator()
	18.4] dataSize.toLong()
	18.5] return trainer.trainingResult
	18.6] .build()
	18.7] .distinct()
	18.8] .toLabeledGraph()
	18.9] add(SEP)
	18.10] createOrLoadModel()
	18.11] PolynomialDecayTracker.builder()
	18.12] consecutive = false
	18.13] executorService.shutdownNow()
	18.14] val offset: Int
	18.15] .shuffled()
	18.16] .readText().lines()
	18.17] WarmUpTracker.builder()
	18.18] } else {
	18.19] add(CLS)


19.] package edu.mcgill.gymfs.agent
	19.0] package edu.mcgill.gymfs.experiments
	19.1] 《package edu.mcgill.gymfs.》inference
	19.2] 《package edu.mcgill.gymfs.》disk
	19.3] 《package edu.mcgill.gymfs》
	19.4] import jetbrains.letsPlot.labe《l.g》gtitle
	19.5] import edu.mcgil《l.g》ymfs.disk.*
	19.6] import com《.g》ithub.jelmerk.knn.SearchResult
	19.7] import jetbrains.datalore.plot.*
	19.8] import ai.hypergraph.kaliningraph.*
	19.9] import jetbrains.letsPlot.intern.*
	19.10] import com.jujutsu.tsne.TSne
	19.11] import com.jujutsu.utils.TSneUtils
	19.12] import org.nield.kotlinstatistics.variance
	19.13] import com.github.jelmerk.knn.*
	19.14] import jetbrains.letsPlot.*
	19.15] import ai.hypergraph.kaliningraph.show
	19.16] import guru.nidi.graphviz.*
	19.17] import kotlin.math.pow
	19.18] import kotlin.system.measureTimeMillis
	19.19] import org.slf4j.LoggerFactory
```
</details>

### Probabilistic Code synthesis

The following command will run the code synthesis demo:

```bash
./gradlew codeSynth -P train=[PATH_TO_TRAINING_DATA]
```

<details>

This should produce something like the following text.

3 symbols / symbol:

```kotlin
fun test = projection be
       val private fun checkbox(): String {
                  }

    fun box(): String {
         as String {
      return "test org.rust
       fun String {
       s
                       }
           va     val box(): String {
                     }
```

3 symbols / symbol:

```python
class Detection_instring else class_componse_source)
           first_list_end]

                           PVOID),
    exception must in not instarted the commension.

                 tokens = 0
            error:
             
       def __name:  line, untile_path)
           no blockThreader sys.get_paracter)
        @rtype:  breated line_filenance',
            if isinstack if not sequeue_size = node):
```

3 symbols / symbol, memory = 2:

```kotlin
val ritingConfig.indefaultResponseExtractory.persDsl {
        */
     * @see [hasNextContentType) }
		fun true): UsertionInjectionInterFunctionse {

    fun result() {
		action that matcher example.order = ReactiveEntityList() {}

	 * @see Request
	inline fun values() = Selections.assure()

             * This bean defining ther the ream()`.
    * @see [list)
		}
	}

    fun val set
    
       @Generate lastImperateBridge
 * @see String, get method instance fun <reified contain await()
      * @params: Map`() {
		val mockRequest = Mocked().buil
```
</details>

### Masked code completion

The following will run the [`CodeCompletion.kt`](src/main/kotlin/edu/mcgill/gymfs/experiments/CodeCompletion.kt) demo:

```bash
./gradlew completeCode
```

### Source Code Transformations

CSTK supports a number of source code transformations for studying the effect on neural language models. Some examples are given below.

#### Synonym renaming

Synonym renaming is provided by [extJWNL](https://github.com/extjwnl/extjwnl). Run the following command:

```bash
./gradlew synonymize
```

<details>

Left column is the original, right column is synonymized:

```
fun VecIndex.knn(v: DoubleArray, i: Int, exact: Boolean = false) =    |    fun VecIndex.knn(v: DoubleArray, i: Int, involve: Boolean = false) =
  if(exact) exactKNNSearch(v, i + 10)                                 |      if(involve) involveKNNSearch(v, i + 10)
  else findNearest(v, i + 10)                                         |      else findNearest(v, i + 10)
    .filter { !it.item().embedding.contentEquals(v) }                 |        .filter { !it.item().embedding.contentEquals(v) }
    .distinctBy { it.item().toString() }.take(i)                      |        .distinctBy { it.item().toString() }.take(i)
============================================================================================================================================
fun VecIndex.exactKNNSearch(vq: DoubleArray, nearestNeighbors: Int) = |    fun VecIndex.exactKNNSearch(vq: DoubleArray, nearestEdge: Int) =
  asExactIndex().findNearest(vq, nearestNeighbors)                    |      asExactIndex().findNearest(vq, nearestEdge)
============================================================================================================================================
  override fun vector(): DoubleArray = embedding                      |      override fun variable(): DoubleArray = embedding
============================================================================================================================================
  override fun dimensions(): Int = embedding.size                     |      override fun mark(): Int = embedding.size
============================================================================================================================================
  override fun toString() = loc.getContext(0)                         |      override fun toWithdraw() = loc.getContext(0)
}                                                                     |    }
============================================================================================================================================
fun main() {                                                          |    fun main() {
  buildOrLoadVecIndex()                                               |      baseOrDepositVecFurnish()
}                                                                     |    }
============================================================================================================================================
fun buildOrLoadKWIndex(                                               |    fun buildOrLoadKWIndex(
  indexFile: File = File(DEFAULT_KNNINDEX_FILENAME),                  |      regulateIncriminate: File = File(DEFAULT_KNNINDEX_FILENAME),
  rootDir: URI = TEST_DIR                                             |      rootDir: URI = TEST_DIR
): KWIndex =                                                          |    ): KWIndex =
  if (!indexFile.exists())                                            |      if (!regulateIncriminate.exists())
    rebuildKWIndex(rootDir).apply { serializeTo(indexFile) }          |        rebuildKWIndex(rootDir).apply { serializeTo(regulateIncriminate) }
  else indexFile.deserializeFrom()                                    |      else regulateIncriminate.deserializeFrom()
============================================================================================================================================
fun main() {                                                          |    fun main() {
  buildOrLoadKWIndex(                                                 |      intensifyOrConcernKWFact(
    indexFile = File(DEFAULT_KWINDEX_FILENAME),                       |        indexFile = File(DEFAULT_KWINDEX_FILENAME),
    rootDir = File("data").toURI()                                    |        rootDir = File("data").toURI()
  )                                                                   |      )
}                                                                     |    }
============================================================================================================================================
fun String.shuffleLines() = lines().shuffled().joinToString("\n")     |    fun String.walkDepression() = lines().shuffled().joinToString("\n")
============================================================================================================================================
fun String.swapPlusMinus() =                                          |    fun String.goQualityMinus() =
  map { if (it == '+') '-' else it }.joinToString("")                 |      map { if (it == '+') '-' else it }.joinToString("")
============================================================================================================================================
fun String.fuzzLoopBoundaries(): String =                             |    fun String.fuzzLoopBoundaries(): String =
  replace(Regex("(for|while)(.*)([0-9]+)(.*)")) { match ->            |      replace(Regex("(for|while)(.*)([0-9]+)(.*)")) { change ->
    match.groupValues.let { it[1] + it[2] +                           |        change.groupValues.let { it[1] + it[2] +
      (it[3].toInt() + (1..3).random()) + it[4] }                     |          (it[3].toInt() + (1..3).random()) + it[4] }
  }                                                                   |      }
============================================================================================================================================
fun String.swapMultilineNoDeps(): String =                            |    fun String.swapMultilineNoDeps(): String =
  lines().chunked(2).map { lines ->                                   |      reenforce().chunked(2).map { reenforce ->
    if (lines.size != 2) return@map lines                             |        if (reenforce.size != 2) return@map reenforce
    val (a, b) = lines.first() to lines.last()                        |        val (a, b) = reenforce.first() to reenforce.last()
    // Same indentation                                               |        // Same indentation
    if (a.trim().length - a.length != b.trim().length - b.length)     |        if (a.trim().length - a.length != b.trim().length - b.length)
      return@map listOf(a, b)                                         |          return@map listOf(a, b)
============================================================================================================================================
fun String.addDeadCode(): String =                                    |    fun String.reckonDeadLabel(): String =
  lines().joinToString("\n") {                                        |      lines().joinToString("\n") {
    if (Math.random() < 0.3) "$it; int deadCode = 2;" else it         |        if (Math.random() < 0.3) "$it; int deadCode = 2;" else it
  }                                                                   |      }
============================================================================================================================================
fun main() = TrainSeq2Seq.runExample()                                |    fun main() = TrainSeq2Seq.contendRepresentation()
============================================================================================================================================
  override fun getData(manager: NDManager): Iterable<Batch> =         |      override fun buyData(manager: NDManager): Iterable<Batch> =
    object: Iterable<Batch>, Iterator<Batch> {                        |        object: Iterable<Batch>, Iterator<Batch> {
      var maskedInstances: List<MaskedInstance> = createEpochData()   |          var maskedInstances: List<MaskedInstance> = createEpochData()
      var idx: Int = batchSize                                        |          var idx: Int = batchSize
============================================================================================================================================
      override fun hasNext(): Boolean = idx < maskedInstances.size    |          override fun bangNext(): Boolean = idx < maskedInstances.size
============================================================================================================================================
  override fun prepare(progress: Progress?) {                         |      override fun prepare(progress: Progress?) {
    // get all applicable files                                       |        // get all applicable files
    parsedFiles = TEST_DIR.allFilesRecursively(FILE_EXT)              |        analyzeAccuse = TEST_DIR.allFilesRecursively(FILE_EXT)
      .map { it.toPath() }                                            |          .map { it.toPath() }
      // read & tokenize them                                         |          // read & tokenize them
      .map { parseFile(it) }                                          |          .map { parseFile(it) }
    // determine dictionary                                           |        // determine dictionary
    dictionary = buildDictionary(countTokens(parsedFiles))            |        dictionary = buildDictionary(countTokens(analyzeAccuse))
  }                                                                   |      }
============================================================================================================================================
  fun getDictionarySize(): Int = dictionary!!.tokens.size             |      fun channeliseDictionaryFiller(): Int = dictionary!!.tokens.size
============================================================================================================================================
    operator fun get(id: Int): String =                               |        operator fun get(id: Int): String =
      if (id >= 0 && id < tokens.size) tokens[id] else UNK            |          if (id >= 0 && id < sign.size) sign[id] else UNK
============================================================================================================================================
    operator fun get(token: String): Int =                            |        operator fun get(sign: String): Int =
      tokenToId.getOrDefault(token, UNK_ID)                           |          signToId.getOrDefault(sign, UNK_ID)
============================================================================================================================================
    fun toTokens(ids: List<Int>): List<String> = ids.map { this[it] } |        fun toSymbol(ids: List<Int>): List<String> = ids.map { this[it] }
============================================================================================================================================
    fun getRandomToken(rand: Random?): String =                       |        fun getRandomToken(rand: Random?): String =
      tokens[rand!!.nextInt(tokens.size)]                             |          disk[rand!!.nextInt(disk.size)]
============================================================================================================================================
    private fun batchFromList(                                        |        private fun batchFromList(
      ndManager: NDManager,                                           |          metalTrainer: NDManager,
      batchData: List<IntArray>                                       |          batchData: List<IntArray>
    ) = ndManager.create(batchData.toTypedArray())                    |        ) = metalTrainer.create(batchData.toTypedArray())
============================================================================================================================================
    private fun batchFromList(                                        |        private fun assemblageFromEnumerate(
      ndManager: NDManager,                                           |          ndManager: NDManager,
      instances: List<MaskedInstance>,                                |          instances: List<MaskedInstance>,
      f: (MaskedInstance) -> IntArray                                 |          f: (MaskedInstance) -> IntArray
    ): NDArray = batchFromList(ndManager, instances.map { f(it) })    |        ): NDArray = assemblageFromEnumerate(ndManager, instances.map { f(it) })
============================================================================================================================================
fun List<Double>.variance() =                                         |    fun List<Double>.variance() =
  average().let { mean -> map { (it - mean).pow(2) } }.average()      |      cypher().let { mean -> map { (it - mean).pow(2) } }.cypher()
============================================================================================================================================
fun euclidDist(f1: DoubleArray, f2: DoubleArray) =                    |    fun geometerDist(f1: DoubleArray, f2: DoubleArray) =
  sqrt(f1.zip(f2) { a, b -> (a - b).pow(2) }.sum())                   |      sqrt(f1.zip(f2) { a, b -> (a - b).pow(2) }.sum())
============================================================================================================================================
fun Array<DoubleArray>.average(): DoubleArray =                       |    fun Array<DoubleArray>.average(): DoubleArray =
  fold(DoubleArray(first().size)) { a, b ->                           |      fold(DoubleArray(first().size)) { a, b ->
    a.zip(b).map { (i, j) -> i + j }.toDoubleArray()                  |        a.zip(b).map { (i, j) -> i + j }.toBidVesture()
  }.map { it / size }.toDoubleArray()                                 |      }.map { it / size }.toBidVesture()
============================================================================================================================================
  override fun distance(u: DoubleArray, v: DoubleArray) =             |      override fun distance(u: DoubleArray, v: DoubleArray) =
    kantorovich(arrayOf(u), arrayOf(v))                               |        kantorovich(standOf(u), standOf(v))
}                                                                     |    }
============================================================================================================================================
fun main() {                                                          |    fun main() {
  val (a, b) = Pair(randomMatrix(400, 768), randomMatrix(400, 768))   |      val (a, b) = Pair(randomArray(400, 768), randomArray(400, 768))
  println(measureTime { println(kantorovich(a, b)) })                 |      println(measureTime { println(kantorovich(a, b)) })
}                                                                     |    }
============================================================================================================================================
    override fun processInput(                                        |        override fun processInput(
      ctx: TranslatorContext,                                         |          ctx: TranslatorContext,
      inputs: Array<String>                                           |          infix: Array<String>
    ): NDList = NDList(                                               |        ): NDList = NDList(
      NDArrays.stack(                                                 |          NDArrays.stack(
        NDList(inputs.map { ctx.ndManager.create(it) })               |            NDList(infix.map { ctx.ndManager.create(it) })
      )                                                               |          )
    )                                                                 |        )
============================================================================================================================================
    override fun getBatchifier(): Batchifier? = null                  |        override fun channelizeBatchifier(): Batchifier? = null
  }                                                                   |      }
}                                                                     |    }
============================================================================================================================================
fun main() {                                                          |    fun main() {
  val answer = BertQaInference.predict()                              |      val satisfy = BertQaInference.predict()
  BertQaInference.logger.info("Answer: {}", answer)                   |      BertQaInference.logger.info("Answer: {}", satisfy)
}                                                                     |    }
============================================================================================================================================
fun URI.extension() = toString().substringAfterLast('.')              |    fun URI.extension() = toRemove().substringAfterLast('.')
fun URI.suffix() = toString().substringAfterLast('/')                 |    fun URI.suffix() = toRemove().substringAfterLast('/')
============================================================================================================================================
  fun getContext(surroundingLines: Int) =                             |      fun getContext(surroundingPipage: Int) =
    uri.allLines().drop((line - surroundingLines).coerceAtLeast(0))   |        uri.allLines().drop((line - surroundingPipage).coerceAtLeast(0))
      .take(surroundingLines + 1).joinToString("\n") { it.trim() }    |          .take(surroundingPipage + 1).joinToString("\n") { it.trim() }
============================================================================================================================================
  fun fileSummary() = toString().substringBeforeLast(':')             |      fun impeachSummary() = toString().substringBeforeLast(':')
}                                                                     |    }
============================================================================================================================================
fun MutableGraph.show(filename: String = "temp") =                    |    fun MutableGraph.show(name: String = "temp") =
  render(Format.PNG).run {                                            |      render(Format.PNG).run {
    toFile(File.createTempFile(filename, ".png"))                     |        toFile(File.createTempFile(name, ".png"))
  }.show()                                                            |      }.show()
============================================================================================================================================
fun <T> List<Pair<T, T>>.toLabeledGraph(                              |    fun <T> List<Pair<T, T>>.toLabeledGraph(
  toVertex: T.() -> LGVertex = { LGVertex(hashCode().toString()) }    |      toExtreme: T.() -> LGVertex = { LGVertex(hashCode().toString()) }
): LabeledGraph =                                                     |    ): LabeledGraph =
  fold(first().first.toVertex().graph) { acc, (s, t) ->               |      fold(first().first.toExtreme().graph) { acc, (s, t) ->
    val (v, w) = s.toVertex() to t.toVertex()                         |        val (v, w) = s.toExtreme() to t.toExtreme()
    acc + LabeledGraph { v - w; w - v }                               |        acc + LabeledGraph { v - w; w - v }
  }                                                                   |      }
============================================================================================================================================
  override fun run() {                                                |      override fun run() {
    printQuery()                                                      |        availablenessAsk()
    graphs.toIntOrNull()?.let { generateGraphs(it) }                  |        graphs.toIntOrNull()?.let { generateGraphs(it) }
  }                                                                   |      }
============================================================================================================================================
fun URI.slowGrep(query: String, glob: String = "*"): Sequence<QIC> =  |    fun URI.slowGrep(ask: String, glob: String = "*"): Sequence<QIC> =
  allFilesRecursively().map { it.toPath() }                           |      allFilesRecursively().map { it.toPath() }
    .mapNotNull { path ->                                             |        .mapNotNull { path ->
      path.read()?.let { contents ->                                  |          path.read()?.let { contents ->
        contents.extractConcordances(".*$query.*")                    |            contents.extractConcordances(".*$ask.*")
          .map { (cxt, idx) -> QIC(query, path, cxt, idx) }           |              .map { (cxt, idx) -> QIC(ask, path, cxt, idx) }
      }                                                               |          }
    }.flatten()                                                       |        }.flatten()
============================================================================================================================================
```
</details>

# Deployment

Need to build fat JAR locally then deploy, CC doesn't like Gradle for some reason.

```bash
./gradlew jar && scp build/libs/gym-fs-fat-1.0-SNAPSHOT.jar breandan@beluga.calculquebec.ca:/home/breandan/projects/def-jinguo/breandan/gym-fs

salloc -t 3:0:0 --account=def-jinguo --gres=gpu:v100:1 --mem=32G --cpus-per-task=24
```

To start, must have Java and Python with PyTorch and HuggingFace:

```bash
export TRANSFORMERS_OFFLINE=1 && \
module load python/3.8 && \
module load java && \
source venv/bin/activate && \
python embedding_server.py microsoft/graphcodebert-base & && \
java -jar gym-fs-fat-1.0-SNAPSHOT.jar | tee logfile.txt
```

# Research Questions

* Can we learn to synthesize a search query which is likely to retrieve results containing relevant information to the local context
* Do good queries contain keywords from the surrounding context? What information sources are the most salient?
* Can we learn a highly compressed index of all artifacts on GitHub for fast offline lookups with just-in-time retrieval?
* What if when we allowed the user to configure the search settings?
  * Clone type (I/II/III/IV)
  * File extension filter
  * Source code context
  * Edge construction
  * Ranking metric
* How do we clearly communicate search result alignment? Concordance++
* What information can be extracted from the search results? Can we adapt information from results into the local context, to suggest e.g. naming, code fixes?

# Libraries

* [Concurrent Trees](https://github.com/npgall/concurrent-trees) - For fast indexing and retrieval.
* [HNSW](https://github.com/jelmerk/hnswlib) - Java library for approximate nearest neighbors search using Hierarchical Navigable Small World graphs
* [java-string-similarity](https://github.com/tdebatty/java-string-similarity) - Implementation of various string similarity and distance algorithms
* [Commons VFS](https://commons.apache.org/proper/commons-vfs/) - Virtual file system for compressed files
* [LearnLib](https://github.com/LearnLib/learnlib) - Java library for automata learning algorithms
* [OR-Tools](https://developers.google.com/optimization/introduction/overview) - Software suite for combinatorial optimization
* [KeyBERT](https://github.com/MaartenGr/KeyBERT) - Minimal keyword extraction using BERT

# Papers

* [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/pdf/1603.09320.pdf), Malkov & Yashunin (2015
* [BERT-kNN: Adding a kNN Search Component to Pretrained Language Models for Better QA](https://arxiv.org/pdf/2005.00766.pdf), Kassner & Schutze (2020)
* [AutoKG: Constructing Virtual Knowledge Graphs from Unstructured Documents for Question Answering](https://arxiv.org/pdf/2008.08995.pdf), Yu et al. (2021)
* [Graph Optimal Transport for Cross-Domain Alignment](http://proceedings.mlr.press/v119/chen20e/chen20e.pdf), Chen et al. (2021)
* [TextRank: Bringing Order into Texts](https://www.aclweb.org/anthology/W04-3252.pdf), Mihalcea and Tarau (2004)
* [From word embeddings to document distances (WMD)](http://proceedings.mlr.press/v37/kusnerb15.pdf#page=3), Kusner et al. (2015)

# Example-centric programming

* [Example-Centric Programming: Integrating Web Search
into the Development Environment](https://hci.stanford.edu/publications/2009/blueprintTR/brandt_blueprint_techreport.pdf), Brandt et al. (2009)
* [Exemplar: A source code search engine for finding highly relevant applications](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5989838), McMillan et al. (2012)

# Symbolic automata

* [Symbolic Automata for Static Specification Mining](https://cseweb.ucsd.edu/~hpeleg/sas2013.pdf) [[slides](https://cseweb.ucsd.edu/~hpeleg/sa_for_msr.pdf)], Peleg et al. (2013)
* [Symbolic Automata](https://pages.cs.wisc.edu/~loris/symbolicautomata.html), D'Antoni et al.

# Grammar Induction

* [Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples](https://arxiv.org/pdf/1711.09576.pdf), Weiss et al. (2018)
* [Enumerating Regular Expressions and Their Languages](https://cs.uwaterloo.ca/~shallit/Papers/ciaa-04.pdf), Lee & Shallit (2004)
* [BLUE*: a Blue-Fringe Procedure for Learning DFA with Noisy Data](https://www.ibisc.univ-evry.fr/~janodet/pub/tjs04.pdf), Sebban et al. (2004)
* [Learning Regular Sets from Queries and Counterexamples](https://omereingold.files.wordpress.com/2017/06/angluin87.pdf), Angluin (1987)

## Automata-based

* [BRICS](https://github.com/cs-au-dk/dk.brics.automaton)
* [LearnLib](https://github.com/Learnlib/learnlib)
* [JFLAP](http://www.jflap.org/)
* [Symbolic automata](https://github.com/lorisdanto/symbolicautomata)
* [SymLearn](https://github.com/PhillipVH/symlearn)

## RE-based

* [frak](https://github.com/noprompt/frak)
* [rgxg](https://github.com/rgxg/rgxg)
* [Grex](https://github.com/pemistahl/grex)
* [RegexGenerator](https://github.com/MaLeLabTs/RegexGenerator)

# Natural Langauge Processing

* [extJWNL](https://github.com/extjwnl/extjwnl)

# Learning to Rank

* [Learning to Rank with Nonsmooth Cost Functions](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf), Burges et al. (2018)
* [Learning to rank papers](https://en.wikipedia.org/wiki/Learning_to_rank#List_of_methods)

# Resources

* [Query Refinement / Relevance models](https://chauff.github.io/documents/ir2017/Query-Refinement-Lecture.pdf#page=24)

# Benchmarking

* [CodeSearchNet](https://github.com/github/CodeSearchNet)
* [OpenMatch](https://github.com/thunlp/OpenMatch)
* [Steps for Evaluating Search Algorithms](https://shopify.engineering/evaluating-search-algorithms) (e.g. MAP, DCG)