rootProject.name = "gym-fs"

pluginManagement.repositories {
  mavenCentral()
  gradlePluginPortal()
  maven("https://oss.sonatype.org/content/repositories/snapshots")
}

sourceControl {
  gitRepository(java.net.URI.create("https://github.com/JetBrains-Research/ast-transformations.git")) {
    producesModule("org.jetbrains.research.ml.ast.transformations:ast-transformations")
  }
}

includeBuild("galoisenne") {
  dependencySubstitution {
    substitute(module("ai.hypergraph:kaliningraph")).using(project(":"))
  }
}