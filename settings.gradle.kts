rootProject.name = "gym-fs"

pluginManagement.repositories {
  mavenCentral()
  gradlePluginPortal()
  maven("https://maven.pkg.jetbrains.space/kotlin/p/kotlin/dev")
}

sourceControl {
  gitRepository(java.net.URI.create("https://github.com/JetBrains-Research/ast-transformations.git")) {
    producesModule("org.jetbrains.research.ml.ast.transformations:ast-transformations")
  }
}

includeBuild("galoisenne")