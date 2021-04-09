# gym-fs

A fast, RL environment for the filesystem.

Stores BPE-compressed files in memory.

Gives an agent the ability to selectively read files.

Interface:

* `Path.read(start, end)` - Returns file chunk at offset.
* `Path.grep(query)` - Returns offsets matching query.

# Usage

Keyword search:

`./gradlew -q trieSearch --args='--query=<QUERY> [--path=<PATH_TO_INDEX>] [--index=<INDEX_FILE>]'`

For example:

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

Nearest neighbor search:

`./gradlew -q knnSearch --args='--query=<QUERY> [--path=<PATH_TO_INDEX>] [--index=<INDEX_FILE>] [--graphs=10]'`

For example:

```
./gradlew -q knnSearch --args='--query="const val MAX_GPUS = 1"'

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

# Semantic Similarity

What does semantic similarity look like?

<details>

```
Nearest nearest neighbors by cumulative similarity

0.] .toLabeledGraph()
	0.0.] .toTypedArray()
	0.1.] .toList()
	0.2.] .asSequence()
	0.3.] .allCodeFragments()
	0.4.] .renderVKG()
	0.5.] .shuffled()
	0.6.] .distinct()
	0.7.] dataSize.toLong()
	0.8.] .readText().lines()
	0.9.] PolynomialDecayTracker.builder()


1.] const val MAX_BATCH = 50
	1.0.] const val MAX_VOCAB = 35000
	1.1.] const val MAX_GPUS = 1
	1.2.] const val EPOCHS = 100000
	1.3.] const val MAX_SEQUENCE_LENGTH = 128
	1.4.] const val BATCH_SIZE = 24
	1.5.] const val CLS = "<cls>"
	1.6.] const val UNK = "<unk>"
	1.7.] const val BERT_EMBEDDING_SIZE = 768
	1.8.] dataSize.toLong()
	1.9.] val targetEmbedding =


2.] const val MAX_VOCAB = 35000
	2.0.] const val MAX_BATCH = 50
	2.1.] const val EPOCHS = 100000
	2.2.] const val MAX_GPUS = 1
	2.3.] const val MAX_SEQUENCE_LENGTH = 128
	2.4.] const val CLS = "<cls>"
	2.5.] const val BATCH_SIZE = 24
	2.6.] const val UNK = "<unk>"
	2.7.] const val MSK = "<msk>"
	2.8.] const val BERT_EMBEDDING_SIZE = 768
	2.9.] dataSize.toLong()


3.] .toTypedArray()
	3.0.] .toLabeledGraph()
	3.1.] .toList()
	3.2.] .asSequence()
	3.3.] .shuffled()
	3.4.] .allCodeFragments()
	3.5.] dataSize.toLong()
	3.6.] .distinct()
	3.7.] .renderVKG()
	3.8.] WarmUpTracker.builder()
	3.9.] PolynomialDecayTracker.builder()


4.] const val EPOCHS = 100000
	4.0.] const val MAX_VOCAB = 35000
	4.1.] const val MAX_BATCH = 50
	4.2.] const val MAX_GPUS = 1
	4.3.] const val MAX_SEQUENCE_LENGTH = 128
	4.4.] const val CLS = "<cls>"
	4.5.] const val BATCH_SIZE = 24
	4.6.] const val UNK = "<unk>"
	4.7.] const val MSK = "<msk>"
	4.8.] const val SEP = "<sep>"
	4.9.] const val BERT_EMBEDDING_SIZE = 768


5.] .toList()
	5.0.] .toTypedArray()
	5.1.] .toLabeledGraph()
	5.2.] .distinct()
	5.3.] .asSequence()
	5.4.] .shuffled()
	5.5.] .readText().lines()
	5.6.] .allCodeFragments()
	5.7.] .show()
	5.8.] .allFilesRecursively()
	5.9.] dataSize.toLong()


6.] .shuffled()
	6.0.] .distinct()
	6.1.] .renderVKG()
	6.2.] .toLabeledGraph()
	6.3.] .show()
	6.4.] .toTypedArray()
	6.5.] .toList()
	6.6.] .asSequence()
	6.7.] .allCodeFragments()
	6.8.] .build()
	6.9.] dataSize.toLong()


7.] const val MAX_GPUS = 1
	7.0.] const val MAX_BATCH = 50
	7.1.] const val MAX_VOCAB = 35000
	7.2.] const val EPOCHS = 100000
	7.3.] const val BATCH_SIZE = 24
	7.4.] const val MAX_SEQUENCE_LENGTH = 128
	7.5.] const val CLS = "<cls>"
	7.6.] dataSize.toLong()
	7.7.] const val BERT_EMBEDDING_SIZE = 768
	7.8.] const val UNK = "<unk>"
	7.9.] const val CODEBERT_CLS_TOKEN = "<s>"


8.] PolynomialDecayTracker.builder()
	8.0.] WarmUpTracker.builder()
	8.1.] PaddingStackBatchifier.builder()
	8.2.] dataSize.toLong()
	8.3.] TrainBertOnCode.runExample()
	8.4.] executorService.shutdownNow()
	8.5.] trainer.metrics = Metrics()
	8.6.] .shuffled()
	8.7.] .toLabeledGraph()
	8.8.] .toTypedArray()
	8.9.] .distinct()
```
</details>

# Deployment

Need to build fat JAR locally then deploy, CC doesn't like Gradle for some reason.

```
./gradlew jar && scp build/libs/gym-fs-fat-1.0-SNAPSHOT.jar breandan@beluga.calculquebec.ca:/home/breandan/projects/def-jinguo/breandan/gym-fs
```

To reindex, first start CodeBERT server, to vectorize the code fragments:

```bash
# Serves vectorized code fragments at http://localhost:8000/?<QUERY>
python codebert_server.py
```

# Resources

* [Concurrent Trees](https://github.com/npgall/concurrent-trees) - For fast indexing and retrieval.
* [Jimfs](https://github.com/google/jimfs) - An in-memory file system for dynamic document parsing.
* [HNSW](https://github.com/jelmerk/hnswlib) - Java library for approximate nearest neighbors search using Hierarchical Navigable Small World graphs
* [java-string-similarity](https://github.com/tdebatty/java-string-similarity) - Implementation of various string similarity and distance algorithms

# Papers

* [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/pdf/1603.09320.pdf), Malkov & Yashunin (2015
* [BERT-kNN: Adding a kNN Search Component to Pretrained Language Models for Better QA](https://arxiv.org/pdf/2005.00766.pdf), Kassner & Schutze (2020)
* [AutoKG: Constructing Virtual Knowledge Graphs from Unstructured Documents for Question Answering](https://arxiv.org/pdf/2008.08995.pdf), Yu et al. (2021)
* [Graph Optimal Transport for Cross-Domain Alignment](http://proceedings.mlr.press/v119/chen20e/chen20e.pdf), Chen et al. (2021)

# Benchmarking

* [CodeSearchNet](https://github.com/github/CodeSearchNet)
* [OpenMatch](https://github.com/thunlp/OpenMatch)
* [Steps for Evaluating Search Algorithms](https://shopify.engineering/evaluating-search-algorithms) (e.g. MAP, DCG)