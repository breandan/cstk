package edu.mcgill.cstk.experiments.rewriting

import com.beust.klaxon.Klaxon
import edu.mcgill.cstk.rewriting.permuteArgumentOrder
import java.io.File
import kotlin.reflect.KFunction


val testInput = """
{
  "repo": "ReactiveX/RxJava", 
  "path": "src/main/java/io/reactivex/Observable.java", 
  "func_name": "Observable.sorted", 
  "original_string": "@CheckReturnValue\n    @SchedulerSupport(SchedulerSupport.NONE)\n    public final Observable<T> sorted(Comparator<? super T> sortFunction) {\n        ObjectHelper.requireNonNull(sortFunction, \"sortFunction is null\");\n        return toList().toObservable().map(Functions.listSorter(sortFunction)).flatMapIterable(Functions.<List<T>>identity());\n    }", 
  "language": "java", 
  "code": "@CheckReturnValue\n    @SchedulerSupport(SchedulerSupport.NONE)\n    public final Observable<T> sorted(Comparator<? super T> sortFunction) {\n        ObjectHelper.requireNonNull(sortFunction, \"sortFunction is null\");\n        return toList().toObservable().map(Functions.listSorter(sortFunction)).flatMapIterable(Functions.<List<T>>identity());\n    }", 
  "code_tokens": ["@", "CheckReturnValue", "@", "SchedulerSupport", "(", "SchedulerSupport", ".", "NONE", ")", "public", "final", "Observable", "<", "T", ">", "sorted", "(", "Comparator", "<", "?", "super", "T", ">", "sortFunction", ")", "{", "ObjectHelper", ".", "requireNonNull", "(", "sortFunction", ",", "\"sortFunction is null\"", ")", ";", "return", "toList", "(", ")", ".", "toObservable", "(", ")", ".", "map", "(", "Functions", ".", "listSorter", "(", "sortFunction", ")", ")", ".", "flatMapIterable", "(", "Functions", ".", "<", "List", "<", "T", ">", ">", "identity", "(", ")", ")", ";", "}"], 
  "transformation": permuteArgument
  "docstring": "Returns an Observable that emits the events emitted by source ObservableSource, in a\nsorted order based on a specified comparison function.\n\n<p>Note that calling {@code sorted} with long, non-terminating or infinite sources\nmight cause {@link OutOfMemoryError}\n\n<dl>\n<dt><b>Scheduler:</b></dt>\n<dd>{@code sorted} does not operate by default on a particular {@link Scheduler}.</dd>\n</dl>\n\n@param sortFunction\na function that compares two items emitted by the source ObservableSource and returns an Integer\nthat indicates their sort order\n@return an Observable that emits the items emitted by the source ObservableSource in sorted order", 
  "docstring_tokens": ["Returns", "an", "Observable", "that", "emits", "the", "events", "emitted", "by", "source", "ObservableSource", "in", "a", "sorted", "order", "based", "on", "a", "specified", "comparison", "function", "."], 
  "sha": "ac84182aa2bd866b53e01c8e3fe99683b882c60e", 
  "url": "https://github.com/ReactiveX/RxJava/blob/ac84182aa2bd866b53e01c8e3fe99683b882c60e/src/main/java/io/reactivex/Observable.java#L12008-L12013", 
  "partition": "test"
}
"""

fun main() {
  File("test.jsonl").transform(String::permuteArgumentOrder)
}

data class CXGSnippet(
  val repo: String,
  val path: String,
  val func_name: String,
  val original_string: String,
  val language: String,
  var code: String,
  var code_tokens: List<String>,
  val docstring: String,
  val docstring_tokens: List<String>,
  val sha: String,
  val url: String,
  val partition: String,
)

fun File.transform(kFunction: KFunction<String>) =
  if (extension != "jsonl") throw Exception("Unsupported format: $extension ")
  else File(nameWithoutExtension + "_${kFunction.name}.jsonl").writeText(
    readLines().joinToString("\n") {
      val result = Klaxon().parse<CXGSnippet>(it)
      result!!.code = result.code.permuteArgumentOrder()
      Klaxon().toJsonString(result).also { println(it) }
    }
  )