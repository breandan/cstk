package edu.berkeley.nlp.util;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class StrUtils
{

	public static <T> String join(List<T> objs) {
		return join(objs, " ");
	}

	public static <T> String join(T[] objs, String delim) {
		if (objs == null) return "";
		return join(Arrays.asList(objs), delim);
	}

	public static <T> String join(List<T> objs, String delim) {
		if (objs == null) return "";
		return join(objs, delim, 0, objs.size());
	}

	public static <T> String join(List<T> objs, String delim, int start, int end) {
		if (objs == null) return "";
		StringBuilder sb = new StringBuilder();
		boolean first = true;
		for (int i = start; i < end; i++) {
			if (!first) sb.append(delim);
			sb.append(objs.get(i));
			first = false;
		}
		return sb.toString();
	}

	public static <T> String join(Collection<T> objs, String delim) {
		if (objs == null) return "";
		StringBuilder sb = new StringBuilder();
		boolean first = true;
		for (T x : objs) {
			if (!first) sb.append(delim);
			sb.append(x);
			first = false;
		}
		return sb.toString();
	}

}
