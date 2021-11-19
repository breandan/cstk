package edu.berkeley.nlp.util;

import java.io.Serializable;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;

/**
 * Maintains a two-way map between a set of objects and contiguous integers from
 * 0 to the number of objects. Use get(i) to look up object i, and
 * indexOf(object) to look up the index of an object.
 * 
 * @author Dan Klein
 */
public class StringIndexer extends AbstractList<String> implements Serializable
{
	private static final long serialVersionUID = -8769544079136550516L;

	List<String> objects;

	StringToIntOpenHashMap indexes;

	/**
	 * Return the object with the given index
	 * 
	 * @param index
	 */
	@Override
	public String get(int index) {
		return objects.get(index);
	}

	/**
	 * Returns the number of objects indexed.
	 */
	@Override
	public int size() {
		return objects.size();
	}

	/**
	 * Returns the index of the given object, or -1 if the object is not present
	 * in the indexer.
	 * 
	 * @param o
	 * @return
	 */
	@Override
	public int indexOf(Object o) {
		if (!(o instanceof String)) return -1;
		int index = indexes.get((String) o);

		return index;
	}

	/**
	 * Add an element to the indexer if not already present. In either case,
	 * returns the index of the given object.
	 * 
	 * @param e
	 * @return
	 */
	public int addAndGetIndex(String e) {
		int index = indexes.get(e);
		if (index >= 0) { return index; }
		//  Else, add
		int newIndex = size();
		objects.add(e);
		indexes.put(e, newIndex);
		return newIndex;
	}

	/**
	 * Add an element to the indexer. If the element is already in the indexer,
	 * the indexer is unchanged (and false is returned).
	 * 
	 * @param e
	 * @return
	 */
	@Override
	public boolean add(String e) {
		return addAndGetIndex(e) == size() - 1;
	}

	public StringIndexer() {
		objects = new ArrayList<String>();
		indexes = new StringToIntOpenHashMap();
	}

}
