package edu.berkeley.nlp.util;

import java.util.Arrays;

/**
 * Open address hash map with linear probing. Maps Strings to int's. Note that
 * int's are assumed to be non-negative, and -1 is returned when a key is not
 * present.
 * 
 * @author adampauls
 * 
 */
public class StringToIntOpenHashMap
{

	private String[] keys = new String[10];

	private int[] values = new int[] { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

	private int size = 0;

	private static final double MAX_LOAD_FACTOR = 0.75;

	public boolean put(String k, int v) {
		if (size / (double) keys.length > MAX_LOAD_FACTOR) {
			rehash();
		}
		return putHelp(k, v, keys, values);

	}

	/**
	 * 
	 */
	private void rehash() {
		String[] newKeys = new String[keys.length * 3 / 2];
		int[] newValues = new int[values.length * 3 / 2];
		Arrays.fill(newValues, -1);
		size = 0;
		for (int i = 0; i < keys.length; ++i) {
			String curr = keys[i];
			if (curr != null) {
				int val = values[i];
				putHelp(curr, val, newKeys, newValues);
			}
		}
		keys = newKeys;
		values = newValues;
	}

	/**
	 * @param k
	 * @param v
	 */
	private boolean putHelp(String k, int v, String[] keyArray, int[] valueArray) {
		int pos = getInitialPos(k, keyArray);
		String curr = keyArray[pos];
		while (curr != null && !curr.equals(k)) {
			pos++;
			if (pos == keyArray.length) pos = 0;
			curr = keyArray[pos];
		}

		valueArray[pos] = v;
		if (curr == null) {
			size++;
			keyArray[pos] = k;
			return true;
		}
		return false;
	}

	/**
	 * @param k
	 * @param keyArray
	 * @return
	 */
	private int getInitialPos(String k, String[] keyArray) {
		int hash = k.hashCode();
		if (hash < 0) hash = -hash;
		int pos = hash % keyArray.length;
		return pos;
	}

	public int get(String k) {
		int pos = getInitialPos(k, keys);
		String curr = keys[pos];
		while (curr != null && !curr.equals(k)) {
			pos++;
			if (pos == keys.length) pos = 0;
			curr = keys[pos];
		}

		return values[pos];
	}

}
