package edu.berkeley.nlp.langmodel.impl;


/**
 * Manages storage of arbitrary values in an NgramMap
 * 
 * @author adampauls
 * 
 * @param <V>
 */
public interface ValueContainer<V>
{

	/**
	 * Adds a new value at the specified offset.
	 * 
	 * @param ngramOrder
	 * @param offset
	 * @param prefixOffset
	 * @param word
	 * @param val
	 * @param suffixOffset
	 */
	public void add(int ngramOrder, long offset, long prefixOffset, int word, V val, long suffixOffset);

	/**
	 * Swaps values at offsets a and b.
	 * 
	 * @param a
	 * @param b
	 * @param ngramOrder
	 */
	public void swap(long a, long b, int ngramOrder);

	/**
	 * Shifts <code>length</code> consecutive values starting at src to the
	 * offset starting at dest. Much like System.arraycopy
	 * 
	 * @param ngramOrder
	 * @param src
	 * @param dest
	 * @param length
	 */
	public void shift(int ngramOrder, long src, long dest, int length);

	/**
	 * Sets internal storage for size for a particular n-gram order
	 * 
	 * @param size
	 * @param ngramOrder
	 */
	public void setSizeAtLeast(long size, int ngramOrder);

	/**
	 * Creates a fresh value container for copying purposes.
	 * 
	 * @return
	 */
	public ValueContainer<V> createFreshValues();

	/**
	 * Gets the value living at a particular offset.
	 * 
	 * @param offset
	 * @param ngramOrder
	 * @return
	 */
	public V getFromOffset(long offset, int ngramOrder);

	/**
	 * Destructively sets internal storage from another object.
	 * 
	 * @param other
	 */
	public void setFromOtherValues(ValueContainer<V> other);

	public void clearStorageAfterCompression(int ngramOrder);

	public void trimAfterNgram(int ngramOrder, long size);

	/**
	 * Final clean up of storage.
	 */
	public void trim();

	/**
	 * Retrieves a stored context (suffix) offset for a n-gram at an offset.
	 * 
	 * @param offset
	 * @param ngramOrder
	 * @return
	 */
	public long getContextOffset(long offset, int ngramOrder);

}