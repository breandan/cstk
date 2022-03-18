package edu.mcgill.cstk.math

import ai.hypergraph.kaliningraph.types.*

fun <A: Any, B: Any> List<Π2<A?, B>>.filterFirstNotNull(): List<Π2<A, B>> =
  mapNotNull {
    when (val fst: A? = it.first) {
      null -> null
      else -> fst pp it.second
    }
  }

fun <A: Any, B: Any> List<Π2<A, B?>>.filterSecondNotNull(): List<Π2<A, B>> =
  mapNotNull {
    when (val s: B? = it.second) {
      null -> null
      else -> it.first pp s
    }
  }

inline fun <T: Any, O: Any> List<T>.joinNotNull(
  others: List<O>,
  on: (a: T, b: O) -> Boolean = { a, b -> a == b }
): List<Π2<T, List<O>>> =
  map { it pp others.filter { o: O -> on(it, o) } }

inline fun <T: Any, O: Any, K: Any> List<T>.joinNotNull(
  others: Map<K, O>, on: (a: T, b: Map<K, O>) -> List<O?>
): List<Π2<T, List<O>>> = map { it pp on(it, others).filterNotNull() }

inline fun <T: Any, O: Any> List<T>.leftJoinFirst(
  others: List<O>,
  on: (a: T, b: O) -> Boolean
): List<Π2<T, O?>> = map { it pp others.firstOrNull { o: O -> on(it, o) } }

inline fun <T: Any, O: Any, K: Any> List<T>.leftJoinFirst(
  others: Map<K, O>, on: (a: T, b: Map<K, O>) -> O?
): List<Π2<T, O?>> = map { it pp on(it, others) }

inline fun <T: Any, O: Any, K: Any> List<T>.innerJoinFirst(
  others: Map<K, O>, on: (a: T, b: Map<K, O>) -> O?
): List<Π2<T, O>> =
  mapNotNull {
    when (val theOther: O? = on(it, others)) {
      null -> null
      else -> it pp theOther
    }
  }

inline fun <T: Any, O: Any> List<T>.flatMapInnerJoin(
  others: List<O>,
  on: (a: T, b: O) -> Boolean = { a, b -> a == b }
): List<Π2<T, O>> =
  flatMap {
    others
      .filter { o: O -> on(it, o) }
      .map { o: O -> it pp o }
  }

inline fun <T: Any, O: Any> List<T>.innerJoinFirst(
  others: List<O>,
  on: (a: T, b: O) -> Boolean = { a, b -> a == b }
): List<Π2<T, O>> =
  mapNotNull {
    when (val theOther: O? = others.firstOrNull { o: O -> on(it, o) }) {
      null -> null
      else -> it pp theOther
    }
  }

inline fun <T: Any, O: Any> List<T>.flatMapLeftJoin(
  others: List<O>,
  on: (a: T, b: O) -> Boolean = { a, b -> a == b }
): List<Π2<T, O?>> =
  flatMap { me ->
    val theOthers: List<O> = others.filter { o: O -> on(me, o) }
    if (theOthers.isEmpty()) {
      listOf(me pp null)
    } else {
      theOthers.map { o -> me pp o }
    }
  }

inline fun <T: Any, O: Any> List<T>.mapLeftJoin(
  others: List<O>,
  on: (a: T, b: O) -> Boolean = { a, b -> a == b }
): List<Π2<T, List<O>>> =
  map { me -> me pp others.filter { o: O -> on(me, o) } }