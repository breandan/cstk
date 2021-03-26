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

`./gradlew -q knnSearch --args='--query=<QUERY> [--path=<PATH_TO_INDEX>] [--index=<INDEX_FILE>]'`

For example:

```
$ ./gradlew -q knnSearch --args='--query="const val MAX_GPUS = 1"'

Searching index of size 1939 for [?]=[const val MAX_GPUS = 1]…

|-----> Original index before reranking by query similarity
|    |-----> Current index after reranking by similarity metric
|    |
503->0.) const val MAX_GPUS = 1
278->1.) const val MAX_BATCH = 50
203->2.) const val FPS = 1000 / 30
498->3.) const val MAX_VOCAB = 35000
132->4.) const val FINAL_EPSILON = 0.0001f
241->5.) const val FILE_EXT = "*.kt"
270->6.) const val MAX_PIPE_COUNT = 30 // 对象池中对象的最大个数
114->7.) private var height = 0
334->8.) private var birdState = 0
 82->9.) rectY + RECT_DESCALE * 2,

Fetched nearest neighbors in 1.354119ms
```

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

# Papers

* [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/pdf/1603.09320.pdf), Malkov & Yashunin (2015
* [BERT-kNN: Adding a kNN Search Component to Pretrained Language Models for Better QA](https://arxiv.org/pdf/2005.00766.pdf), Kassner & Schutze (2020)