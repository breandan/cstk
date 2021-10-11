package edu.mcgill.cstk.math


fun <T, U> cartProd(c1: Iterable<T>, c2: Iterable<U>): List<Pair<T, U>> =
  c1.flatMap { lhsElem -> c2.map { rhsElem -> lhsElem to rhsElem } }


fun <A: Any, B: Any> List<Pair<A?, B>>.filterFirstNotNull(): List<Pair<A, B>> =
  mapNotNull {
    val fst: A? = it.first
    when (fst) {
      null -> null
      else -> fst to it.second
    }
  }

fun <A: Any, B: Any> List<Pair<A, B?>>.filterSecondNotNull(): List<Pair<A, B>> =
  mapNotNull {
    val s: B? = it.second
    when (s) {
      null -> null
      else -> it.first to s
    }
  }

inline fun <T: Any, O: Any> List<T>.joinNotNull(
  others: List<O>,
  on: (a: T, b: O) -> Boolean = { a, b -> a == b }
): List<Pair<T, List<O>>> =
  map { it: T -> it to others.filter { o: O -> on(it, o) } }

inline fun <T: Any, O: Any, K: Any> List<T>.joinNotNull(
  others: Map<K, O>, on: (a: T, b: Map<K, O>) -> List<O?>
): List<Pair<T, List<O>>> = map { it to on(it, others).filterNotNull() }

inline fun <T: Any, O: Any> List<T>.leftJoinFirst(
  others: List<O>,
  on: (a: T, b: O) -> Boolean
): List<Pair<T, O?>> = map { it to others.firstOrNull { o: O -> on(it, o) } }

inline fun <T: Any, O: Any, K: Any> List<T>.leftJoinFirst(
  others: Map<K, O>, on: (a: T, b: Map<K, O>) -> O?
): List<Pair<T, O?>> = map { it to on(it, others) }

inline fun <T: Any, O: Any, K: Any> List<T>.innerJoinFirst(
  others: Map<K, O>, on: (a: T, b: Map<K, O>) -> O?
): List<Pair<T, O>> =
  mapNotNull {
    when (val theOther: O? = on(it, others)) {
      null -> null
      else -> it to theOther
    }
  }

inline fun <T: Any, O: Any> List<T>.flatMapInnerJoin(
  others: List<O>,
  on: (a: T, b: O) -> Boolean = { a, b -> a == b }
): List<Pair<T, O>> =
  flatMap {
    others
      .filter { o: O -> on(it, o) }
      .map { o: O -> it to o }
  }

inline fun <T: Any, O: Any> List<T>.innerJoinFirst(
  others: List<O>,
  on: (a: T, b: O) -> Boolean = { a, b -> a == b }
): List<Pair<T, O>> =
  mapNotNull {
    when (val theOther: O? = others.firstOrNull { o: O -> on(it, o) }) {
      null -> null
      else -> it to theOther
    }
  }

inline fun <T: Any, O: Any> List<T>.flatMapLeftJoin(
  others: List<O>,
  on: (a: T, b: O) -> Boolean = { a, b -> a == b }
): List<Pair<T, O?>> =
  flatMap { me ->
    val theOthers: List<O> = others.filter { o: O -> on(me, o) }
    if (theOthers.isEmpty()) {
      listOf(me to null)
    } else {
      theOthers.map { o -> me to o }
    }
  }

inline fun <T: Any, O: Any> List<T>.mapLeftJoin(
  others: List<O>,
  on: (a: T, b: O) -> Boolean = { a, b -> a == b }
): List<Pair<T, List<O>>> =
  map { me -> me to others.filter { o: O -> on(me, o) } }