import org.gradle.api.JavaVersion.VERSION_11
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
  val kotlinVersion = "1.5.0-M1"
  kotlin("jvm") version kotlinVersion
  id("com.github.ben-manes.versions") version "0.38.0"
//  kotlin("plugin.serialization") version kotlinVersion
}

group = "com.github.breandan"
version = "1.0-SNAPSHOT"

repositories {
  mavenCentral()
  maven("https://dl.bintray.com/kotlin/kotlin-datascience")
}

dependencies {
  implementation(kotlin("stdlib"))
//  implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.1.0")
  implementation("com.google.jimfs:jimfs:1.2")
  implementation("com.googlecode.concurrent-trees:concurrent-trees:2.6.1")

  implementation("ai.djl:api:0.10.0")
  implementation("org.slf4j:slf4j-simple:1.7.30")
//  implementation("ai.djl.mxnet:mxnet-engine:0.10.0")
//  implementation("ai.djl.mxnet:mxnet-native-cu102mkl:1.7.0-backport")

//  implementation("ai.djl:examples:0.6.0")

  implementation("ai.djl.sentencepiece:sentencepiece:0.10.0")
  implementation("ai.djl.fasttext:fasttext-engine:0.10.0")
  implementation("ai.djl:model-zoo:0.10.0")
  implementation("commons-cli:commons-cli:1.4")
  implementation("ai.djl.tensorflow:tensorflow-engine:0.10.0")
  implementation("ai.djl.tensorflow:tensorflow-native-cu101:2.3.1")

//  implementation("ai.djl:model-zoo:0.10.0")
  implementation("ai.djl.mxnet:mxnet-model-zoo:0.10.0")

  val hnswlibVersion = "0.0.46"
  implementation("com.github.jelmerk:hnswlib-core:$hnswlibVersion")
  implementation("com.github.jelmerk:hnswlib-utils:$hnswlibVersion")

  val multikVersion = "0.0.1"
  implementation("org.jetbrains.kotlinx:multik-api:$multikVersion")
  implementation("org.jetbrains.kotlinx:multik-default:$multikVersion")
  implementation("com.robrua.nlp:easy-bert:1.0.3")

  implementation("info.debatty:java-string-similarity:2.0.0")

  implementation("com.github.ajalt.clikt:clikt:3.1.0")
}

tasks {
  register("trieSearch", JavaExec::class) {
    main = "edu.mcgill.gymfs.disk.TrieSearchKt"
    classpath = sourceSets["main"].runtimeClasspath
  }

  register("knnSearch", JavaExec::class) {
    main = "edu.mcgill.gymfs.disk.KNNSearchKt"
    classpath = sourceSets["main"].runtimeClasspath
  }

  register("trainBert", JavaExec::class) {
    main = "edu.mcgill.gymfs.agent.BertTrainerKt"
    classpath = sourceSets["main"].runtimeClasspath
  }

  register("trainBird", JavaExec::class) {
    main = "com.kingyu.rlbird.ai.agent.TrainBirdKt"
    classpath = sourceSets["main"].runtimeClasspath
  }

  withType<KotlinCompile> {
    kotlinOptions {
      languageVersion = "1.5"
      apiVersion = "1.5"
      jvmTarget = VERSION_11.toString()
    }
  }

  withType<Jar> {
    manifest.attributes["Main-Class"] = "edu.mcgill.gymfs.agent.BertTrainerKt"

    from(configurations.compileClasspath.get().files
      .map { if (it.isDirectory) it else zipTree(it) })
    archiveBaseName.set("${project.name}-fat")
  }
}