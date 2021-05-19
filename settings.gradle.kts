rootProject.name = "gym-fs"

includeBuild("markovian") {
  dependencySubstitution {
    substitute(module("com.github.breandan:markovian")).with(project(":"))
  }
}