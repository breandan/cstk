import org.gradle.api.JavaVersion.VERSION_1_8
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
  kotlin("jvm") version "1.5.0-M1"
  id("com.github.ben-manes.versions") version "0.38.0"
}

group = "com.github.breandan"
version = "1.0-SNAPSHOT"

repositories.mavenCentral()

dependencies {
  implementation(kotlin("stdlib"))
  implementation("com.google.jimfs:jimfs:1.2")
  implementation("com.googlecode.concurrent-trees:concurrent-trees:2.6.1")

  implementation("ai.djl:api:0.10.0")
  implementation("org.slf4j:slf4j-simple:1.7.30")
  implementation("ai.djl.mxnet:mxnet-engine:0.10.0")
  implementation("ai.djl.mxnet:mxnet-native-cu102mkl:1.7.0-backport")

//  implementation("ai.djl:examples:0.6.0")

  implementation("ai.djl.sentencepiece:sentencepiece:0.10.0")
  implementation("ai.djl.fasttext:fasttext-engine:0.10.0")
  implementation("ai.djl:model-zoo:0.10.0")
  implementation("commons-cli:commons-cli:1.4")

  implementation("com.github.ajalt.clikt:clikt:3.1.0")
}

tasks {
  register("grep", JavaExec::class) {
    main = "edu.mcgill.gymfs.EnvironmentKt"
    classpath = sourceSets["main"].runtimeClasspath
  }

  register("train", JavaExec::class) {
    main = "edu.mcgill.gymfs.BertTrainerKt"
    classpath = sourceSets["main"].runtimeClasspath
  }

  withType<KotlinCompile> {
    kotlinOptions {
      languageVersion = "1.5"
      apiVersion = "1.5"
      jvmTarget = VERSION_1_8.toString()
    }
  }

  withType<Jar> {
    manifest {
      attributes["Main-Class"] = "edu.mcgill.gymfs.BertTrainerKt"
    }

    from(configurations.compileClasspath.get().files
      .map { if (it.isDirectory) it else zipTree(it) })
    archiveBaseName.set("${project.name}-fat")
  }
}