import de.undercouch.gradle.tasks.download.Download
import org.gradle.api.JavaVersion.VERSION_11
import org.gradle.api.file.DuplicatesStrategy.EXCLUDE

plugins {
  val kotlinVersion = "1.5.0"
  kotlin("jvm") version kotlinVersion
  id("com.github.ben-manes.versions") version "0.38.0"
//  kotlin("plugin.serialization") version kotlinVersion
  id("de.undercouch.download") version "4.1.1"
}

group = "com.github.breandan"
version = "1.0-SNAPSHOT"

repositories {
  mavenCentral()
  maven("https://jitpack.io")
  maven("https://jetbrains.bintray.com/lets-plot-maven")
}

dependencies {
  implementation(kotlin("stdlib-jdk8"))
  implementation("org.jetbrains.kotlin:kotlin-reflect:1.5.0")
  implementation(libs.cct)

  implementation(libs.djl)
  implementation(libs.slf4j)
  implementation(libs.djlmxnet)
  implementation(libs.djlmxnn)

//  implementation(libs.djltf)
//  implementation(libs.djltfn)
//  implementation("ai.djl:examples:0.6.0")

  implementation(libs.slf4j)
  implementation(libs.sentencepiece)
  implementation(libs.fasttext)
  implementation(libs.modelzoo)

  implementation("ai.djl.mxnet:mxnet-model-zoo:0.11.0")

  val hnswlibVersion = "0.0.46"
  implementation("com.github.jelmerk:hnswlib-core:$hnswlibVersion")
  implementation("com.github.jelmerk:hnswlib-utils:$hnswlibVersion")

  val multikVersion = "0.0.1"
  implementation("org.jetbrains.kotlinx:multik-api:$multikVersion")
  implementation("org.jetbrains.kotlinx:multik-default:$multikVersion")

  implementation("com.robrua.nlp:easy-bert:1.0.3")

  implementation("org.kohsuke:github-api:1.128")

  implementation("info.debatty:java-string-similarity:2.0.0")

  implementation("com.github.ajalt.clikt:clikt:3.1.0")

  implementation("org.jetbrains.lets-plot-kotlin:lets-plot-kotlin-api:1.3.0")

  implementation("com.github.breandan.T-SNE-Java:tsne:master-SNAPSHOT")

  implementation("org.nield:kotlin-statistics:1.2.1")

  implementation("com.github.breandan:kaliningraph:0.1.6")

  // https://github.com/LearnLib/learnlib
  implementation("de.learnlib.distribution:learnlib-distribution:0.16.0")
  // https://github.com/LearnLib/automatalib
  implementation("net.automatalib.distribution:automata-distribution:0.10.0")
  // https://github.com/cs-au-dk/dk.brics.automaton
  implementation("dk.brics:automaton:1.12-1")

  // For retrieving dataset from GitHub
  implementation("org.kohsuke:github-api:1.128")

  implementation("org.apache.commons:commons-compress:1.20")
  implementation("org.apache.commons:commons-vfs2:2.8.0")
}

tasks {
  register("getGrex", Download::class) {
    onlyIf { !File("grex").exists() }
    val name = "grex-v1.2.0-x86_64-unknown-linux-musl.tar.gz"
    src("https://github.com/pemistahl/grex/releases/download/v1.2.0/$name")
    dest(File(name))
    overwrite(false)

    doLast {
      copy {
        from(tarTree(resources.gzip(name)))
        into(projectDir)
      }
    }
  }

  mapOf(
    "trieSearch" to "edu.mcgill.gymfs.disk.KWSearchKt",
    "knnSearch" to "edu.mcgill.gymfs.disk.KNNSearchKt",
    "cloneRepos" to "edu.mcgill.gymfs.github.CloneReposKt",
    "filterRepos" to "edu.mcgill.gymfs.github.FilterReposKt",
    "trainBert" to "edu.mcgill.gymfs.agent.BertTrainerKt",
    "indexKW" to "edu.mcgill.gymfs.indices.KWIndexKt",
    "indexKNN" to "edu.mcgill.gymfs.indices.VecIndexKt",
    "testRegex" to "edu.mcgill.gymfs.experiments.TestRegexKt",
    "compareMetrics" to "edu.mcgill.gymfs.experiments.CompareMetricsKt",
    "nearestNeighbors" to "edu.mcgill.gymfs.experiments.NearestNeighborsKt",
  ).forEach { (cmd, mainClass) ->
    register(cmd, JavaExec::class) {
      main = mainClass
      classpath = sourceSets["main"].runtimeClasspath
    }
  }

  compileKotlin {
    kotlinOptions.jvmTarget = VERSION_11.toString()
    kotlinOptions.freeCompilerArgs += "-Xuse-experimental=kotlin.Experimental"
  }

  jar {
    manifest.attributes["Main-Class"] = "edu.mcgill.gymfs.agent.BertTrainerKt"

    from(configurations.compileClasspath.get().files
      .filter { it.extension != "pom" }
      .map { if (it.isDirectory) it else zipTree(it) })

    duplicatesStrategy = EXCLUDE
    exclude("META-INF/*.DSA")
    exclude("META-INF/*.RSA")
    exclude("META-INF/*.SF")
    archiveBaseName.set("${project.name}-fat")
  }
}