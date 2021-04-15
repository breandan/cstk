import org.gradle.api.JavaVersion.VERSION_11
import org.gradle.api.file.DuplicatesStrategy.EXCLUDE
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
  val kotlinVersion = "1.5.0-RC"
  kotlin("jvm") version kotlinVersion
  id("com.github.ben-manes.versions") version "0.38.0"
//  kotlin("plugin.serialization") version kotlinVersion
}

group = "com.github.breandan"
version = "1.0-SNAPSHOT"

repositories {
  mavenCentral()
  maven("https://jitpack.io")
  maven("https://jetbrains.bintray.com/lets-plot-maven")
}

dependencies {
  implementation(kotlin("stdlib"))
//  implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.1.0")
  implementation(libs.jimfs)
  implementation(libs.cct)

  implementation(libs.djl)
  implementation(libs.slf4j)
  implementation(libs.djlmxnet)
  implementation(libs.djlmxnn)

//  implementation(libs.djltf)
//  implementation(libs.djltfn)
//  implementation("ai.djl:examples:0.6.0")

  implementation(libs.jimfs)
  implementation(libs.slf4j)
  implementation(libs.sentencepiece)
  implementation(libs.fasttext)
  implementation(libs.modelzoo)

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
  implementation("org.deeplearning4j:deeplearning4j:1.0.0-beta7")
  implementation("org.deeplearning4j:deeplearning4j-modelimport:1.0.0-beta7")

  implementation("org.jetbrains.lets-plot-kotlin:lets-plot-kotlin-api:1.3.0")
  implementation("com.github.breandan.T-SNE-Java:tsne:master-SNAPSHOT")

  implementation("org.nield:kotlin-statistics:1.2.1")

  implementation("com.github.breandan:kotlingrad:0.4.2")
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

  compileKotlin {
    kotlinOptions.jvmTarget = VERSION_11.toString()
  }

  jar {
    manifest.attributes["Main-Class"] = "edu.mcgill.gymfs.agent.BertTrainerKt"

    from(configurations.compileClasspath.get().files
      .filter { it.extension != "pom" }
      .map { if (it.isDirectory) it else zipTree(it) })

    duplicatesStrategy = EXCLUDE
    archiveBaseName.set("${project.name}-fat")
  }
}