package edu.berkeley.nlp.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Dan Klein
 */
public class CollectionUtils
{
	public static <E extends Comparable<E>> List<E> sort(Collection<E> c) {
		List<E> list = new ArrayList<E>(c);
		Collections.sort(list);
		return list;
	}

	public static <E> List<E> sort(Collection<E> c, Comparator<E> r) {
		List<E> list = new ArrayList<E>(c);
		Collections.sort(list, r);
		return list;
	}

	public static <K, V> void addToValueList(Map<K, List<V>> map, K key, V value) {
		List<V> valueList = map.get(key);
		if (valueList == null) {
			valueList = new ArrayList<V>();
			map.put(key, valueList);
		}
		valueList.add(value);
	}

	public static <K, V> List<V> getValueList(Map<K, List<V>> map, K key) {
		List<V> valueList = map.get(key);
		if (valueList == null) return Collections.emptyList();
		return valueList;
	}

	public static <E> List<E> iteratorToList(Iterator<E> iterator) {
		List<E> list = new ArrayList<E>();
		while (iterator.hasNext()) {
			list.add(iterator.next());
		}
		return list;
	}

	public static <E> Set<E> union(Set<? extends E> x, Set<? extends E> y) {
		Set<E> union = new HashSet<E>();
		union.addAll(x);
		union.addAll(y);
		return union;
	}

	/**
	 * Convenience method for constructing lists on one line. Does type
	 * inference:
	 * <code>List<String> args = makeList("-length", "20","-parser","cky");</code>
	 * 
	 * @param <T>
	 * @param elems
	 * @return
	 */
	public static <T> List<T> makeList(T... elems) {
		List<T> list = new ArrayList<T>();
		for (T elem : elems) {
			list.add(elem);
		}
		return list;
	}

	public static long sum(long[] a) {
		if (a == null) { return 0; }
		long result = 0;
		for (int i = 0; i < a.length; i++) {
			result += a[i];
		}
		return result;
	}

	/**
	 * Wraps an iterator as an iterable
	 * 
	 * @param <T>
	 * @param it
	 * @return
	 */
	public static <T> Iterable<T> iterable(final Iterator<T> it) {
		return new Iterable<T>()
		{
			boolean used = false;

			public Iterator<T> iterator() {
				if (used) throw new RuntimeException("One use iterable");
				used = true;
				return it;
			}
		};
	}

	public static long[] copyOf(long[] a, int length) {
		long[] ret = new long[length];
		System.arraycopy(a, 0, ret, 0, Math.min(ret.length, a.length));
		return ret;
	}

	public static int[] copyOf(int[] a, int length) {
		int[] ret = new int[length];
		System.arraycopy(a, 0, ret, 0, Math.min(ret.length, a.length));
		return ret;
	}

	public static double[] copyOf(double[] a, int length) {
		double[] ret = new double[length];
		System.arraycopy(a, 0, ret, 0, Math.min(ret.length, a.length));
		return ret;
	}

	public static int[] copyOfRange(int[] a, int from, int to) {
		int[] ret = new int[to - from];
		System.arraycopy(a, from, ret, 0, ret.length);
		return ret;
	}

	public static void fill(boolean[][] a, boolean b) {
		for (boolean[] c : a) {
			if (c != null) Arrays.fill(c, b);
		}
	}

	public static void fill(int[][] a, int i) {
		for (int[] c : a) {
			if (c != null) Arrays.fill(c, i);
		}
	}
}
