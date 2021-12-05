package edu.mcgill.cstk.crawler

import edu.mcgill.cstk.disk.*
import edu.mcgill.cstk.experiments.defaultTokenizer
import edu.mcgill.cstk.nlp.*
import spoon.Launcher
import spoon.reflect.declaration.*
import java.io.File

fun main() {
  //collectLengthStats()
  collectSubtypeStats()
}

fun collectSubtypeStats() =
  DATA_DIR.allFilesRecursively("tgz", false).mapNotNull {
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
      type.simpleName +
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

fun collectLengthStats() {
  println("total lines, total tokens, avg line len, len comments, len code")
  DATA_DIR.allFilesRecursively()
    .allMethods()
    .forEach { (method, uri) ->
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