rootProject.name = "gym-fs"

pluginManagement.repositories {
  mavenCentral()
  gradlePluginPortal()
}

sourceControl {
  gitRepository(java.net.URI.create("https://github.com/JetBrains-Research/ast-transformations.git")) {
    producesModule("org.jetbrains.research.ml.ast.transformations:ast-transformations")
  }
}

includeBuild("galoisenne")