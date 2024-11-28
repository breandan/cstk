import de.undercouch.gradle.tasks.download.Download
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
  kotlin("jvm") version "2.1.0"
  id("com.github.ben-manes.versions") version "0.51.0"
  id("de.undercouch.download") version "5.6.0"
  id("com.gradleup.shadow") version "8.3.3"
}

group = "com.github.breandan"
version = "1.0-SNAPSHOT"

repositories {
  mavenCentral()
  mavenLocal()
  maven("https://packages.jetbrains.team/maven/p/astminer/astminer")
  maven("https://jitpack.io")
}

dependencies {
  // String index
  implementation("com.googlecode.concurrent-trees:concurrent-trees:2.6.1")

  implementation("org.slf4j:slf4j-simple:2.1.0-alpha1")

//  implementation("ai.djl.tensorflow:tensorflow-engine:0.12.0")
//  implementation("ai.djl.tensorflow:tensorflow-native-cu101:2.3.1")
//  implementation("ai.djl:examples:0.6.0")

  val djlVersion = "0.31.0"
  implementation("ai.djl:api:$djlVersion")
  implementation("ai.djl.mxnet:mxnet-engine:$djlVersion")
  implementation("ai.djl.mxnet:mxnet-native-cu102mkl:1.9.1")
  implementation("ai.djl.fasttext:fasttext-engine:$djlVersion")
  implementation("ai.djl.sentencepiece:sentencepiece:$djlVersion")
  implementation("ai.djl.mxnet:mxnet-model-zoo:$djlVersion")
  implementation("ai.djl:model-zoo:$djlVersion")
  implementation("ai.djl.huggingface:tokenizers:$djlVersion")

  // Vector embedding index
  val hnswlibVersion = "1.1.2"
  implementation("com.github.jelmerk:hnswlib-core:$hnswlibVersion")
  implementation("com.github.jelmerk:hnswlib-utils:$hnswlibVersion")

  val multikVersion = "0.2.3"
  implementation("org.jetbrains.kotlinx:multik-core:$multikVersion")
  implementation("org.jetbrains.kotlinx:multik-default:$multikVersion")

  // String comparison metrics
  implementation("info.debatty:java-string-similarity:2.0.0")

  // CLI parser
  implementation("com.github.ajalt.clikt:clikt:5.0.1")

  implementation("com.beust:klaxon:5.6")

  // Source code transformation
//  implementation("com.github.h0tk3y.betterParse:better-parse:0.4.2")
  val openrwVersion = "8.41.0"
  implementation("org.openrewrite:rewrite-java:$openrwVersion")
  runtimeOnly("org.openrewrite:rewrite-java-11:$openrwVersion")

  val smileVersion = "4.0.0"
  implementation("com.github.haifengl:smile-kotlin:$smileVersion")
  implementation("com.github.haifengl:smile-core:$smileVersion")

  // https://github.com/LearnLib/learnlib
//  implementation("de.learnlib.distribution:learnlib-distribution:0.16.0")
  // https://github.com/LearnLib/automatalib
//  implementation("net.automatalib.distribution:automata-distribution:0.10.0")

//  https://github.com/lorisdanto/symbolicautomata
//  implementation("com.github.lorisdanto.symbolicautomata:0da3f79677")

//  https://github.com/tech-srl/prime
//  implementation("com.github.tech-srl:prime:5fae8f309f")

  // Clustering for automata extraction
  implementation("org.tribuo:tribuo-clustering-kmeans:4.3.1")

  // RegEx to DFA conversion
  // https://github.com/cs-au-dk/dk.brics.automaton
  implementation("dk.brics:automaton:1.12-4")

  // Querying and filtering data from GitHub
  implementation("org.kohsuke:github-api:1.324")
  // Querying and filtering data from GitLab
  implementation("org.gitlab4j:gitlab4j-api:6.0.0-rc.5")

  implementation("org.jetbrains.lets-plot:platf-awt-jvm:4.4.1")
  implementation("org.jetbrains.lets-plot:lets-plot-kotlin-jvm:4.9.2")

  // Read compressed repositories downloaded from GitHub
  implementation("org.apache.commons:commons-compress:1.27.0")
  implementation("org.apache.commons:commons-vfs2:2.9.0")

  // Constraint minimization for Kantorovich-Rubenstein distance
  val ortoolsVersion = "9.11.4210"
  implementation("com.google.ortools:ortools-java:$ortoolsVersion")
  // AArch64 support? https://github.com/google/or-tools/issues/716
  // Darwin/M1 support? https://github.com/google/or-tools/issues/2332
  implementation("com.google.ortools:ortools-linux-x86-64:$ortoolsVersion")

  // Wordnet synonym service
  implementation("net.sf.extjwnl:extjwnl:2.0.5")
  implementation("net.sf.extjwnl:extjwnl-data-wn31:1.2")

  // Pretty-printing string diffs
  implementation("io.github.java-diff-utils:java-diff-utils:4.15")

  // Simulate a browser for scraping JS XML content
  implementation("net.sourceforge.htmlunit:htmlunit:2.70.0")

  // Evaluation metrics for information retrieval
  // https://github.com/qcri/EvaluationMetrics/tree/master/src/main/java/qa/qf/qcri/iyas/evaluation/ir
  implementation("com.github.qcri:EvaluationMetrics:62ff519478")

  // DFA to RegEx conversion
  // https://github.com/LearnLib/learnlib/issues/75
  // http://www.jflap.org/modules/ConvertedFiles/DFA%20to%20Regular%20Expression%20Conversion%20Module.pdf
  // https://github.com/LakshmiAntin/JFLAPEnhanced/blob/cbb1e6a52f44c826fcb082c85cba9e5f09dcdb33/gui/action/ArdenLemma.java
  // implementation("com.github.citiususc:jflap-lib:1.3")

  // Software metrics
//  implementation("com.github.rodhilton:jasome:0.6.8-alpha")
//  implementation("io.joern:javasrc2cpg_2.13:0.0.5")

  // Graph visualization
  implementation("guru.nidi:graphviz-java:0.18.1")
  implementation("guru.nidi:graphviz-kotlin:0.18.1")

  implementation("ai.hypergraph:kaliningraph") {
    //  exclude(group = "org.jetbrains.kotlin", module = "kotlin-stdlib")
//  exclude(group = "org.jetbrains.kotlin", module = "kotlin-stdlib-common")
//  exclude(group = "org.jetbrains.kotlin", module = "kotlin-reflect")
//    exclude(group = "guru.nidi", module = "graphviz-kotlin")
    exclude(group = "org.graalvm.js", module = "js")
    exclude(group = "org.jetbrains.kotlinx", module = "kotlinx-coroutines-core")
    exclude(group = "org.jetbrains.kotlinx", module = "kotlinx-html-jvm")
//          exclude(group = "org.jetbrains.kotlinx", module = "multik-core")
//          exclude(group = "org.jetbrains.kotlinx", module = "multik-default")
    exclude(group = "org.jetbrains.lets-plot", module = "lets-plot-kotlin-jvm")
//    exclude(group = "org.apache.datasketches", module = "datasketches")
//    exclude(group = "org.apache.datasketches", module = "datasketches-java")
    exclude(group = "ca.umontreal.iro.simul", module = "ssj")
    exclude(group = "org.sosy-lab", module = "common")
    exclude(group = "org.sosy-lab", module = "java-smt")
    exclude(group = "org.sosy-lab", module = "javasmt-solver-mathsat5")
  }
  //implementation("org.apache.datasketches:datasketches-memory:3.0.1")
  //implementation("org.apache.datasketches:datasketches-java:6.1.0")
  implementation("org.apache.datasketches:datasketches-java:6.1.1")
  implementation("io.github.vovak:astminer:0.9.0")
  implementation("com.github.ben-manes.caffeine:caffeine:3.1.8")

  // Source Code Transformations
  implementation("fr.inria.gforge.spoon:spoon-core:11.1.1-beta-8")

//  implementation("com.theokanning.openai-gpt3-java:api:0.12.0")
  implementation("com.aallam.openai:openai-client:3.8.2")

  // Common statistical tests
  implementation("org.hipparchus:hipparchus-stat:3.1")

//  implementation("io.github.danielnaczo:python3parser:1.0.4")
  implementation("org.antlr:antlr4:4.13.2")

  implementation("com.diffplug.spotless:spotless-lib:3.0.0.RC2")

  implementation("org.jetbrains.kotlin:kotlin-compiler:2.1.0")
  implementation(files("libs/kotlin-grammar-tools-0.1-43.jar"))
}

configurations.all {
  resolutionStrategy {
    force("org.antlr:antlr4-runtime:4.7.1")
    force("org.antlr:antlr4-tool:4.7.1")
  }
}

val jvmPreviewFlags = listOf("--enable-preview", "--add-modules=jdk.incubator.vector")

tasks {
  mapOf(
    "trieSearch" to "edu.mcgill.cstk.disk.KWSearchKt",
    "knnSearch" to "edu.mcgill.cstk.disk.KNNSearchKt",
    "trainBert" to "edu.mcgill.cstk.agent.BertTrainerKt",
    "indexKW" to "edu.mcgill.cstk.indices.KWIndexKt",
    "indexKNN" to "edu.mcgill.cstk.indices.VecIndexKt",
    "querySynth" to "edu.mcgill.cstk.experiments.search.DFAExtractionKt",
    "compareMetrics" to "edu.mcgill.cstk.experiments.search.CompareMetricsKt",
    "compareCodeTxs" to "edu.mcgill.cstk.experiments.rewriting.CodeTxComparisonKt",
    "testCodeTxs" to "edu.mcgill.cstk.rewriting.CodeTxTestKt",
    "nearestNeighbors" to "edu.mcgill.cstk.experiments.search.NearestNeighborsKt",
    "codeSynth" to "edu.mcgill.cstk.experiments.probing.CodeSynthesisKt",
    "allTasks" to "edu.mcgill.cstk.experiments.probing.AllTasksKt",
    "completeCode" to "edu.mcgill.cstk.experiments.probing.CodeCompletionKt",
    "compilerTest" to "edu.mcgill.cstk.experiments.probing.CompileTestingKt",
    "completeSyntax" to "edu.mcgill.cstk.experiments.probing.SyntaxCompletionKt",
    "completeDoc" to "edu.mcgill.cstk.experiments.probing.DocCompletionKt",
    "synthCode" to "edu.mcgill.cstk.experiments.probing.CodeSynthesisKt",
    "varMisuse" to "edu.mcgill.cstk.experiments.probing.VariableMisuseKt",
    "codeMetrics" to "edu.mcgill.cstk.math.CodeMetricsKt",
    "code2Vec" to "edu.mcgill.cstk.experiments.search.Code2VecKt",
    "vizCodeEmbed" to "edu.mcgill.cstk.experiments.search.VizCodeEmbeddingsKt",
    "astMiner" to "edu.mcgill.cstk.experiments.search.ASTMinerKt",
    "spoon" to "edu.mcgill.cstk.experiments.rewriting.SpoonTestKt",
    "sampleRepos" to "edu.mcgill.cstk.crawler.SampleReposKt",
    "localizedSyntaxRepair" to "edu.mcgill.cstk.experiments.repair.LocalizedSyntaxRepairKt",
    "syntheticSyntaxRepair" to "edu.mcgill.cstk.experiments.repair.SyntheticSyntaxRepairKt",
    "organicSyntaxRepair" to "edu.mcgill.cstk.experiments.repair.OrganicSyntaxRepairKt",
    "javaSemanticRepair" to "edu.mcgill.cstk.experiments.repair.JavaSemanticRepairKt",
    "collectSummaryStats" to "edu.mcgill.cstk.experiments.repair.CollectSummaryStatisticsKt",
    "kotlinSemanticRepair" to "edu.mcgill.cstk.experiments.repair.KotlinSemanticRepairKt",
    "kotlinStatementRepair" to "edu.mcgill.cstk.experiments.repair.KotlinStatementRepairKt",
    "pythonStatementRepair" to "edu.mcgill.cstk.experiments.repair.PythonStatementRepairKt",
    "pythonSnippetRepair" to "edu.mcgill.cstk.experiments.repair.PythonSnippetRepairKt",
    "contextualRepair" to "edu.mcgill.cstk.experiments.repair.PythonContextualRepairKt",
    "pythonBarHillelRepair" to "edu.mcgill.cstk.experiments.repair.PythonBarHillelRepairKt",
    "javaBarHillelRepair" to "edu.mcgill.cstk.experiments.repair.JavaBarHillelRepairKt",
    "aggregateAndPlot" to "edu.mcgill.cstk.experiments.repair.AggregateAndPlotKt",
    "extractRepairSamples" to "edu.mcgill.cstk.experiments.repair.ExtractRepairSamplesKt",
    "promptRepair" to "edu.mcgill.cstk.experiments.repair.RepairPromptingKt",
    "cloneRepos" to "edu.mcgill.cstk.crawler.CloneReposKt",
    "collectStats" to "edu.mcgill.cstk.crawler.CollectStatsKt",
    "transformJson" to "edu.mcgill.cstk.experiments.TransformCodeXGlueDataKt",
    "tokenize" to "edu.mcgill.cstk.utils.TokenizerKt",
    "probeLLaMA" to "edu.mcgill.cstk.experiments.probing.ProbeLLaMAKt",
  ).forEach { (cmd, main) ->
    register(cmd, JavaExec::class) {
      mainClass = main
      minHeapSize = "4g"
      maxHeapSize = "100g"

      if (cmd == "probeLLaMA")  {
        standardInput = System.`in`
        jvmArgs = jvmPreviewFlags
      }
      classpath = sourceSets["main"].runtimeClasspath
      outputs.upToDateWhen { false }
    }
  }

  compileJava {
    options.compilerArgs.addAll(jvmPreviewFlags)
  }

  compileKotlin {
    compilerOptions.jvmTarget = JvmTarget.JVM_21
    dependsOn("downloadKotlinGrammarTools")
  }

  shadowJar {
    manifest.attributes["Main-Class"] =
//      "edu.mcgill.cstk.experiments.probing.ProbeLLaMAKt"
      "edu.mcgill.cstk.experiments.repair.PythonBarHillelRepairKt"
//      "edu.mcgill.cstk.experiments.CodeCompletionKt"
//      "edu.mcgill.cstk.experiments.DocCompletionKt"
    // Use this to generate the training dataset
//  manifest.attributes["Main-Class"] = "edu.mcgill.cstk.crawler.CloneReposKt"
    isZip64 = true
    archiveFileName = "${project.name}-fat-${project.version}.jar"
    exclude("**/bifi/**", "**/py150/**", "**/seq2parse/**", "**/bifi.zip", "**/stack_overflow.zip")
    minimize()
  }

  register<Download>("downloadKotlinGrammarTools") {
    val version = "0.1-43"
    src("https://github.com/Kotlin/grammar-tools/releases/download/v$version/kotlin-grammar-tools-$version.jar")
    dest("$rootDir/libs/kotlin-grammar-tools-$version.jar")
    overwrite(false)
  }
}
