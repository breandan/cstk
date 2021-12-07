package edu.mcgill.cstk.crawler

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.experiments.defaultTokenizer
import edu.mcgill.cstk.nlp.*
import spoon.Launcher
import spoon.reflect.declaration.*
import java.io.File
import java.net.URI

fun main() {
  //DATA_DIR.collectMethodStats()
  // https://gist.github.com/breandan/afac9ef7e7f2d7f0302f8a0f5926fe4d
  //DATA_DIR.collectSubtypeStats()
  // https://gist.github.com/breandan/cdb780ae883b7e49de1596fe0de96849
  GCODE_DIR.collectLineStats { it.extension() in setOf("java") }
}

fun URI.collectSubtypeStats() =
  allFilesRecursively("tgz", false).mapNotNull {
    println(it)
    val f = File(it).unzip()
    val launcher = Launcher()
    val uniqueTypes = mutableSetOf<String>()
    f.toURI().allFilesRecursively().forEach {
      try {
        if (uniqueTypes.add(it.suffix())) launcher.addInputResource(it.path)
      } catch (e: Exception) { /*e.printStackTrace()*/ }
    }

    try { launcher.buildModel() } catch(e: Exception) { return@mapNotNull null }

    launcher.model.allTypes.filterNotNull().joinToString("") { type ->
      val supertypes = type.superTypes().filterNotNull()
        .map { it.simpleName + if (it.isClass) "(C)" else "(I)" }
      val allMembers = type.allMembers()
      val (allFields, allMethods) = allMembers.let { (f, m) -> f.size to m.size }
      val (fields, methods) = type.fields.size to type.methods.size
      "\t" + type.simpleName +
        (if (supertypes.isNotEmpty()) "<:${supertypes.joinToString(",", "{", "}")}" else "") +
        " (local: $fields fields, $methods methods) " +
        (if(supertypes.isEmpty())"" else "/ (local+inherited: $allFields fields, $allMethods methods)") +
        "\n"
    }
  }.take(10).forEach { println("$it\n") }

fun CtType<*>?.superTypes() =
  if (this == null) emptyList()
  else (superclass?.let { listOf(it.typeDeclaration) } ?: emptyList()) +
    superInterfaces.mapNotNull { kotlin.runCatching { it?.typeDeclaration }.getOrNull() }

// Returns number of inherited fields and methods
fun CtType<*>?.allMembers(
  maxHeight: Int = 10,
  superTypes: List<CtType<*>> = superTypes()
): Pair<Set<CtField<*>>, Set<CtMethod<*>>> =
  if (this == null || superTypes.isEmpty() || maxHeight == 0)
    emptySet<CtField<*>>() to emptySet()
  else (fields.toSet() to methods.toSet()) +
    superTypes.fold(emptySet<CtField<*>>() to emptySet()) { p, it ->
      p + it.allMembers(maxHeight - 1)
    }

operator fun <A, B> Pair<Set<A>, Set<B>>.plus(other: Pair<Set<A>, Set<B>>): Pair<Set<A>, Set<B>> =
  first + other.first to second + other.second

fun URI.collectLineStats(filter: (URI) -> Boolean) {
  println("repo, total files, total lines, lines without comments and blanks")
  allFilesRecursively(readCompressed = false).filter { filter(it) }
    .groupBy { it.path.substringAfter("gcode/").substringBefore("/") }
    .entries.forEach { (repoName, uris) ->
      val (totalLines, codeLines) =
        uris.flatMap { it.contents()?.lines() ?: emptyList() }
          .fold(0 to 0) { (a, b), it ->
            a + 1 to b + if (it.isLineACommentOrEmpty()) 0 else 1
          }

      println("$repoName, ${uris.size}, $totalLines, $codeLines")
    }
}

fun String.isLineACommentOrEmpty(commentPrefixes: Set<String> = setOf("//", "* ", "/*")) =
  trim().let{ it.length <= 1 || it.take(2) in commentPrefixes }

fun URI.collectMethodStats() {
  println("total lines, total tokens, avg line len, len comments, len code")
  allFilesRecursively().allMethods().forEach { (method, uri) ->
      try {
        val string = method.toString()
        val lines = string.lines()
        println(
          "" +
            lines.size + ", " +
            defaultTokenizer.tokenize(string).size +
            lines.size + ", " +
            lines.map { defaultTokenizer.tokenize(it).size }.average()
              .toInt() + ", " +
            defaultTokenizer.tokenize(method.docComment ?: "").size + ", " +
            defaultTokenizer.tokenize(method.body?.toString() ?: "").size
        )
      } catch (_: Exception) {}
    }
}