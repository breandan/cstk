package edu.berkeley.nlp.langmodel.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.FastPriorityQueue;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.PriorityQueue;

abstract class LmValueContainer<V extends Comparable<V>> implements ValueContainer<V>, Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 964277160049236607L;

	int[][] valueRanksUncompressed = new int[10][];

	long[] valueRanksUncompressedSizes = new long[10];

	int[][] valueRanksCompressed;

	transient Indexer<V> countIndexer;

	transient List<Long>[] countCounter;

	int[][] contextOffsets;

	protected int countCutoffForHuffman;

	protected final boolean storePrefixIndexes;

	protected final boolean useHuffman;

	final int valueRadix;

	@SuppressWarnings("unchecked")
	public LmValueContainer(Indexer<V> countIndexer, int valueRadix, int countCutoffForHuffman, boolean useHuffman, boolean storePrefixIndexes) {
		this.countCutoffForHuffman = countCutoffForHuffman;
		this.useHuffman = useHuffman;
		this.valueRadix = valueRadix;
		this.countIndexer = countIndexer;
		this.storePrefixIndexes = storePrefixIndexes;
		if (storePrefixIndexes) contextOffsets = new int[6][];
		countCounter = new List[6];
		for (int i = 0; i < countCounter.length; ++i)
			countCounter[i] = new ArrayList<Long>();

	}

	public void swap(long a_, long b_, int ngramOrder) {
		final int[] arrayForNgram = valueRanksUncompressed[ngramOrder];
		int a = (int) a_;
		int b = (int) b_;
		int temp = arrayForNgram[a];
		assert temp >= 0;
		final int val = arrayForNgram[b];
		assert val >= 0;
		arrayForNgram[a] = val;
		arrayForNgram[b] = temp;
	}

	public void add(int ngramOrder, long offset, long prefixOffset, int word, final V val_, long suffixOffset) {

		V val = val_;
		if (val == null) val = getDefaultVal();

		setSizeAtLeast(100000, ngramOrder);
		final int indexOfCounts = countIndexer.addAndGetIndex(val);
		final int size = countCounter[ngramOrder].size();
		if (indexOfCounts >= size) {
			for (int i = size; i <= indexOfCounts; ++i) {
				countCounter[ngramOrder].add(0L);
			}
		}
		countCounter[ngramOrder].set(indexOfCounts, countCounter[ngramOrder].get(indexOfCounts) + 1);
		if (contextOffsets != null) {
			if (offset >= contextOffsets[ngramOrder].length)
				contextOffsets[ngramOrder] = Arrays.copyOf(contextOffsets[ngramOrder],
					(int) Math.max(offset + 1, contextOffsets[ngramOrder].length * 3 / 2 + 1));
			contextOffsets[ngramOrder][(int) offset] = (int) suffixOffset;
		}
		if (offset >= valueRanksUncompressed[ngramOrder].length) {
			valueRanksUncompressed[ngramOrder] = Arrays.copyOf(valueRanksUncompressed[ngramOrder],
				(int) Math.max(offset + 1, valueRanksUncompressed[ngramOrder].length * 3 / 2 + 1));
		}
		valueRanksUncompressed[ngramOrder][(int) offset] = indexOfCounts;
		valueRanksUncompressedSizes[ngramOrder] = Math.max(offset + 1, valueRanksUncompressedSizes[ngramOrder]);
	}

	abstract protected V getDefaultVal();

	abstract protected void storeCounts(Indexer<V> countIndexer_, int[] indexToSortedIndexMap, int ngramOrder);

	public void shift(int ngramOrder, long src, long dest, int length) {
		if (length == 0) return;
		if (src == dest) return;
		System.arraycopy(valueRanksUncompressed[ngramOrder], (int) src, valueRanksUncompressed[ngramOrder], (int) dest, length);
	}

	public void setSizeAtLeast(long size, int ngramOrder) {
		if (ngramOrder >= valueRanksUncompressed.length) {
			valueRanksUncompressed = Arrays.copyOf(valueRanksUncompressed, valueRanksUncompressed.length * 2);
			if (contextOffsets != null) contextOffsets = Arrays.copyOf(contextOffsets, contextOffsets.length * 2);
		}

		if (valueRanksUncompressed[ngramOrder] == null) {
			valueRanksUncompressed[ngramOrder] = new int[(int) (size + 1)];
			if (contextOffsets != null) contextOffsets[ngramOrder] = new int[(int) (size + 1)];
		}
		if (size > valueRanksUncompressedSizes[ngramOrder]) {
			valueRanksUncompressed[ngramOrder] = Arrays.copyOf(valueRanksUncompressed[ngramOrder], (int) size);
			if (contextOffsets != null) contextOffsets[ngramOrder] = Arrays.copyOf(contextOffsets[ngramOrder], (int) size);
		}
	}

	public V getFromOffset(long index, int ngramOrder) {
		if (valueRanksUncompressed[ngramOrder] != null) {
			int currCountRank = valueRanksUncompressed[ngramOrder][(int) index];
			return countIndexer.get(currCountRank);
		} else {
			int countRank = valueRanksCompressed[ngramOrder][(int) index];
			return getCount(countRank, ngramOrder);
		}
	}

	public long getContextOffset(long index, int ngramOrder) {
		return contextOffsets == null ? -1L : contextOffsets[ngramOrder][(int) index];
	}

	public void setFromOtherValues(ValueContainer<V> other) {
		LmValueContainer<V> o = (LmValueContainer<V>) other;
		this.valueRanksCompressed = o.valueRanksCompressed;
		this.valueRanksUncompressed = o.valueRanksUncompressed;
		this.countIndexer = o.countIndexer;
		this.contextOffsets = o.contextOffsets;
	}

	abstract protected V getCount(int rank, int ngramOrder);

	public void clearStorageAfterCompression(int ngramOrder) {
		valueRanksCompressed = null;
	}

	public void trimAfterNgram(int ngramOrder, long size) {
		valueRanksUncompressed[ngramOrder] = Arrays.copyOf(valueRanksUncompressed[ngramOrder], (int) size);
		if (contextOffsets != null) contextOffsets[ngramOrder] = Arrays.copyOf(contextOffsets[ngramOrder], (int) size);
		if (valueRanksCompressed == null) valueRanksCompressed = new int[6][];
		int[] indexToSortedIndexMap = getIndexToSortedIndexMap(ngramOrder);
		valueRanksCompressed[ngramOrder] = new int[(int) size];
		for (long i = 0; i < size; ++i) {
			if (i >= valueRanksUncompressedSizes[ngramOrder]) break;
			int currCountIndex = valueRanksUncompressed[ngramOrder][(int) i];
			int sortedIndex = indexToSortedIndexMap[currCountIndex];
			if (sortedIndex >= 0) valueRanksCompressed[ngramOrder][(int) i] = sortedIndex;
		}
		//		valueRanksCompressed[ngramOrder].trim();

		storeCounts(countIndexer, indexToSortedIndexMap, ngramOrder);
		//		Logger.logss("Found " + countIndexer.size() + " unique counts");
		valueRanksUncompressed[ngramOrder] = null;
		if (useHuffman) {
			Counter<Integer> sortedCounter = new Counter<Integer>();
			for (int i = 0; i < countCounter[ngramOrder].size(); ++i) {
				sortedCounter.setCount(indexToSortedIndexMap[i], countCounter[ngramOrder].get(i));
			}
			countCounter[ngramOrder] = null;
		}
		countCounter[ngramOrder] = null;
	}

	/**
	 * @param ngramOrder
	 * @return
	 */
	private int[] getIndexToSortedIndexMap(int ngramOrder) {
		PriorityQueue<Integer> sortedIndexes = new FastPriorityQueue<Integer>();
		//		Counter<Integer> countCounterCounter = new Counter<Integer>();
		for (int i = 0; i < countCounter[ngramOrder].size(); ++i) {
			sortedIndexes.setPriority(i, countCounter[ngramOrder].get(i));
		}

		int[] indexToSortedIndexMap = new int[countIndexer.size()];
		Arrays.fill(indexToSortedIndexMap, -1);
		int l = 0;
		for (int i : CollectionUtils.iterable(sortedIndexes)) {
			indexToSortedIndexMap[i] = l++;
		}
		return indexToSortedIndexMap;
	}

	public void trim() {
		countCounter = null;
		countIndexer = null;

	}

}