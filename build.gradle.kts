import org.gradle.api.JavaVersion.VERSION_11

plugins {
  val kotlinVersion = "1.5.30-RC"
  kotlin("jvm") version kotlinVersion
  id("com.github.ben-manes.versions") version "0.39.0"
//  kotlin("plugin.serialization") version kotlinVersion
  id("de.undercouch.download") version "4.1.2"
}

group = "com.github.breandan"
version = "1.0-SNAPSHOT"

repositories {
  mavenCentral()
  maven("https://jitpack.io")
  maven("https://oss.sonatype.org/content/repositories/snapshots")
}

dependencies {
  implementation(platform(kotlin("bom")))
  implementation(kotlin("stdlib-jdk8"))
  implementation(kotlin("reflect"))

  // String index
  implementation("com.googlecode.concurrent-trees:concurrent-trees:2.6.1")

  implementation("org.slf4j:slf4j-simple:1.7.32")

//  implementation("ai.djl.tensorflow:tensorflow-engine:0.12.0")
//  implementation("ai.djl.tensorflow:tensorflow-native-cu101:2.3.1")
//  implementation("ai.djl:examples:0.6.0")

  val djlVersion = "0.12.0"
  implementation("ai.djl:api:$djlVersion")
  implementation("ai.djl.mxnet:mxnet-engine:$djlVersion")
  implementation("ai.djl.mxnet:mxnet-native-cu102mkl:1.7.0-backport")
  implementation("ai.djl.fasttext:fasttext-engine:$djlVersion")
  implementation("ai.djl.sentencepiece:sentencepiece:$djlVersion")
  implementation("ai.djl.mxnet:mxnet-model-zoo:$djlVersion")
  implementation("ai.djl:model-zoo:$djlVersion")

  // Vector embedding index
  val hnswlibVersion = "0.0.46"
  implementation("com.github.jelmerk:hnswlib-core:$hnswlibVersion")
  implementation("com.github.jelmerk:hnswlib-utils:$hnswlibVersion")

  val multikVersion = "0.0.1"
  implementation("org.jetbrains.kotlinx:multik-api:$multikVersion")
  implementation("org.jetbrains.kotlinx:multik-default:$multikVersion")

  // String comparison metrics
  implementation("info.debatty:java-string-similarity:2.0.0")

  // CLI parser
  implementation("com.github.ajalt.clikt:clikt:3.2.0")

  // Source code transformation
//  implementation("fr.inria.gforge.spoon:spoon-core:9.1.0-beta-20")
//  implementation("com.github.h0tk3y.betterParse:better-parse:0.4.2")
  val openrwVersion = "7.11.1"
  implementation("org.openrewrite:rewrite-java:$openrwVersion")
  runtimeOnly("org.openrewrite:rewrite-java-11:$openrwVersion")

//  implementation("org.jetbrains.lets-plot-kotlin:lets-plot-kotlin:1.3.0")
  implementation("com.github.breandan.T-SNE-Java:tsne:master-SNAPSHOT")

  implementation("com.github.breandan:kaliningraph:0.1.7")
  implementation("com.github.breandan:markovian:master-SNAPSHOT")

  // https://github.com/LearnLib/learnlib
  implementation("de.learnlib.distribution:learnlib-distribution:0.16.0")
  // https://github.com/LearnLib/automatalib
  implementation("net.automatalib.distribution:automata-distribution:0.10.0")

//  https://github.com/lorisdanto/symbolicautomata
//  implementation("com.github.lorisdanto.symbolicautomata:0da3f79677")

//  https://github.com/tech-srl/prime
//  implementation("com.github.tech-srl:prime:5fae8f309f")

  // Clustering for automata extraction
  implementation("org.tribuo:tribuo-clustering-kmeans:4.1.0")

  // RegEx to DFA conversion
  // https://github.com/cs-au-dk/dk.brics.automaton
  implementation("dk.brics:automaton:1.12-3")

  // Querying and filtering data from GitHub
  implementation("org.kohsuke:github-api:1.132")

  implementation("org.jetbrains.lets-plot:lets-plot-kotlin-jvm:3.0.2")

  // Read compressed repositories downloaded from GitHub
  implementation("org.apache.commons:commons-compress:1.21")
  implementation("org.apache.commons:commons-vfs2:2.9.0")

  // Constraint minimization for Kantorovich-Rubenstein distance
  val ortoolsVersion = "9.0.9048"
  implementation("com.google.ortools:ortools-java:$ortoolsVersion")
  // AArch64 support? https://github.com/google/or-tools/issues/716
  // Darwin/M1 support? https://github.com/google/or-tools/issues/2332
  implementation("com.google.ortools:ortools-linux-x86-64:$ortoolsVersion")

  // Synonym service
  implementation("net.sf.extjwnl:extjwnl:2.0.3")
  implementation("net.sf.extjwnl:extjwnl-data-wn31:1.2")

  // DFA to RegEx conversion
  // https://github.com/LearnLib/learnlib/issues/75
  // http://www.jflap.org/modules/ConvertedFiles/DFA%20to%20Regular%20Expression%20Conversion%20Module.pdf
  // https://github.com/LakshmiAntin/JFLAPEnhanced/blob/cbb1e6a52f44c826fcb082c85cba9e5f09dcdb33/gui/action/ArdenLemma.java
  // implementation("com.github.citiususc:jflap-lib:1.3")
}

tasks {
  mapOf(
    "trieSearch" to "edu.mcgill.gymfs.disk.KWSearchKt",
    "knnSearch" to "edu.mcgill.gymfs.disk.KNNSearchKt",
    "cloneRepos" to "edu.mcgill.gymfs.github.CloneReposKt",
    "filterRepos" to "edu.mcgill.gymfs.github.FilterReposKt",
    "trainBert" to "edu.mcgill.gymfs.agent.BertTrainerKt",
    "indexKW" to "edu.mcgill.gymfs.indices.KWIndexKt",
    "indexKNN" to "edu.mcgill.gymfs.indices.VecIndexKt",
    "querySynth" to "edu.mcgill.gymfs.experiments.DFAExtractorKt",
    "compareMetrics" to "edu.mcgill.gymfs.experiments.CompareMetricsKt",
    "compareCodeTxs" to "edu.mcgill.gymfs.experiments.CodeTxComparisonKt",
    "testCodeTxs" to "edu.mcgill.gymfs.experiments.CodeTxTestKt",
    "nearestNeighbors" to "edu.mcgill.gymfs.experiments.NearestNeighborsKt",
    "codeSynth" to "edu.mcgill.gymfs.experiments.CodeSynthKt",
    "completeCode" to "edu.mcgill.gymfs.experiments.CodeCompletionKt",
  ).forEach { (cmd, main) ->
    register(cmd, JavaExec::class) {
      mainClass.set(main)
      classpath = sourceSets["main"].runtimeClasspath
    }
  }

  compileKotlin {
    kotlinOptions.jvmTarget = VERSION_11.toString()
    kotlinOptions.freeCompilerArgs += "-Xuse-experimental=kotlin.Experimental"
  }

// Compile fatjar for Compute Canada, doesn't like Gradle
//  jar {
//    manifest.attributes["Main-Class"] = "edu.mcgill.gymfs.experiments.VisualizeTSNEKt"
//
//    from(configurations.compileClasspath.get().files
//      .filter { it.extension != "pom" }
//      .map { if (it.isDirectory) it else zipTree(it) })
//
//    duplicatesStrategy = EXCLUDE
//    exclude("META-INF/*.DSA")
//    exclude("META-INF/*.RSA")
//    exclude("META-INF/*.SF")
//    archiveBaseName.set("${project.name}-fat")
//  }
}