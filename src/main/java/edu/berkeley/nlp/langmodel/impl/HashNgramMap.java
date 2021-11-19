package edu.berkeley.nlp.langmodel.impl;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.langmodel.impl.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.mt.decoder.MurmurHash;

public class HashNgramMap<T> extends NgramMap<T>
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private static final int PRIME = 31;

	private static final int NUM_INDEX_BITS = 36;

	private static final int WORD_BIT_OFFSET = NUM_INDEX_BITS;

	private static final int INDEX_OFFSET = 0;

	private static final long SUFFIX_MASK = mask(NUM_INDEX_BITS, INDEX_OFFSET);

	transient private long[] cachedLastIndex = new long[6];

	transient private int[][] cachedLastSuffix = new int[6][];

	private final boolean cacheSuffixes;

	private final boolean quadraticProbing;

	private static long cachedSuffixHits = 0L;

	private static long cachedSuffixMisses = 0L;

	private static long numProbes = 0;

	private static long numQueries = 0;

	@SuppressWarnings("ucd")
	private static long numQueryCalls = 0;

	@SuppressWarnings("ucd")
	private static long numQueryProbes = 0;

	private static final class HashMap implements Serializable
	{

		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;

		final long[] keys;

		final int[] noWordKeys;

		final long[] wordRangesLow;

		final long[] wordRangesHigh;

		//		LargeLongArray valsAndNexts;

		long numFilled = 0;

		private final boolean quadraticProbing;

		private static final int EMPTY_KEY = -1;

		HashMap(long initialCapacity, long[] numNgramsForEachWord, boolean quadraticProbing) {
			this.quadraticProbing = quadraticProbing;
			if (numNgramsForEachWord != null) {
				keys = null;
				wordRangesLow = new long[numNgramsForEachWord.length];
				wordRangesHigh = new long[numNgramsForEachWord.length];
				long currStart = 0;
				for (int w = 0; w < numNgramsForEachWord.length; ++w) {
					wordRangesLow[w] = currStart;
					currStart += numNgramsForEachWord[w];
					wordRangesHigh[w] = currStart;

				}
				noWordKeys = new int[(int) currStart];
				Arrays.fill(noWordKeys, EMPTY_KEY);
			} else {
				wordRangesLow = wordRangesHigh = null;
				noWordKeys = null;
				keys = new long[(int) initialCapacity];

				//			keys.fill(EMPTY_KEY, initialCapacity);
				Arrays.fill(keys, EMPTY_KEY);
			}
			//			valsAndNexts = new LargeLongArray(initialCapacity);
			//			valsAndNexts.fill(-1L, initialCapacity);
			numFilled = 0;
		}

		private long getKey(long index) {
			return noWordKeys == null ? keys[(int) index] : noWordKeys[(int) index];
		}

		private final int getNext(final int i_, final int start, final int end, final int numProbesSoFar) {
			int i = i_;
			if (quadraticProbing) {
				i += numProbesSoFar;
			} else {
				++i;
			}
			if (i >= end) i = start;
			return i;
		}

		public long put(long index, long putKey) {
			final int firstWordOfNgram = wordOf(putKey);
			final int rangeStart = wordRangesLow == null ? 0 : (int) wordRangesLow[firstWordOfNgram];
			final int rangeEnd = wordRangesHigh == null ? keys.length : (int) wordRangesHigh[firstWordOfNgram];
			long searchKey = getKey(index);
			int i = (int) index;
			int numProbesHere = 1;
			while (searchKey != EMPTY_KEY && searchKey != putKey) {

				i = getNext(i, rangeStart, rangeEnd, numProbesHere);
				++numProbesHere;
				if (numProbesHere > 10000) {
					@SuppressWarnings("unused")
					int x = 5;
				}
				searchKey = getKey(i);
			}

			if (searchKey == EMPTY_KEY) setKey(i, putKey);
			numProbes += numProbesHere;
			++numQueries;
			//			setValue(i, value);

			numFilled++;

			return i;
		}

		private void setKey(long index, long putKey) {
			if (noWordKeys == null) {
				assert keys[(int) index] == EMPTY_KEY;
				keys[(int) index] = putKey;
			} else {
				assert noWordKeys[(int) index] == EMPTY_KEY;
				final int suffixIndex = (int) prefixOffsetOf(putKey);
				assert suffixIndex >= 0;
				noWordKeys[(int) index] = suffixIndex;
			}
		}

		public final long getIndex(final int[] ngram, final long index, final int startPos, final int endPos, final HashMap[] maps) {
			if (keys == null) return getIndexImplicity(ngram, index, startPos, endPos, maps);
			final long[] localKeys = keys;
			final int firstWordOfNgram = ngram[endPos - 1];
			final int keysLength = keys.length;
			final int rangeStart = 0;
			final int rangeEnd = keysLength;
			assert index >= rangeStart;
			assert index < rangeEnd;
			int i = (int) index;
			int num = 1;
			while (true) {
				final long searchKey = localKeys[i];
				int next = getNext(i, rangeStart, rangeEnd, num);
				num++;
				if (searchKey == EMPTY_KEY) {//
					return -1L;
				}
				if (firstWordOfNgram == wordOf(searchKey) && suffixEquals(prefixOffsetOf(searchKey), ngram, startPos, endPos - 1, maps)) { //
					return i;
				}
				i = next;

			}
		}

		public final long getIndex(final long key, final long suffixIndex, final int firstWord, final long index) {
			if (keys == null) return getIndexImplicity(suffixIndex, firstWord, index);
			final long[] localKeys = keys;
			final int keysLength = keys.length;
			final int rangeStart = 0;
			final int rangeEnd = keysLength;
			assert index >= rangeStart;
			assert index < rangeEnd;
			int i = (int) index;
			int num = 1;
			while (true) {
				if (i == rangeEnd) i = rangeStart;
				final long searchKey = localKeys[i];
				if (searchKey == key) { //
					return i;
				}
				if (searchKey == EMPTY_KEY) {//
					return -1L;
				}
				if (quadraticProbing) {
					i += num;
					num++;
				} else {
					++i;
				}
				if (i >= rangeEnd) i = rangeStart;

			}
		}

		public final long getIndexImplicity(long suffixIndex, int firstWordOfNgram, final long index) {
			final int[] localKeys = noWordKeys;
			final int rangeStart = (int) wordRangesLow[firstWordOfNgram];
			final int rangeEnd = (int) wordRangesHigh[firstWordOfNgram];
			assert index >= rangeStart;
			assert index < rangeEnd;
			int i = (int) index;
			int num = 1;
			while (true) {
				if (i == rangeEnd) i = rangeStart;
				final int searchKey = localKeys[i];
				if (searchKey == suffixIndex) {//
					return i;
				}
				if (searchKey == EMPTY_KEY) {//
					return -1L;
				}
				if (quadraticProbing) {
					i += num;
					num++;
				} else {
					++i;
				}
				if (i >= rangeEnd) i = rangeStart;

			}
		}

		public final long getIndexImplicity(final int[] ngram, final long index, final int startPos, final int endPos, final HashMap[] maps) {
			final int[] localKeys = noWordKeys;
			final int firstWordOfNgram = ngram[endPos - 1];
			final int rangeStart = (int) wordRangesLow[firstWordOfNgram];
			final int rangeEnd = (int) wordRangesHigh[firstWordOfNgram];
			assert index >= rangeStart;
			assert index < rangeEnd;
			int i = (int) index;
			int num = 1;
			while (true) {
				if (i == rangeEnd) i = rangeStart;
				final int searchKey = localKeys[i];
				if (searchKey == EMPTY_KEY) {//
					return -1L;
				}
				if (implicitSuffixEquals(searchKey, ngram, startPos, endPos - 1, maps)) { //
					return i;
				}
				if (quadraticProbing) {
					i += num;
					num++;
				} else {
					++i;
				}
				if (i >= rangeEnd) i = rangeStart;

			}
		}

		public long getCapacity() {
			return noWordKeys == null ? keys.length : noWordKeys.length;
		}

		public double getLoadFactor() {
			return (double) numFilled / getCapacity();
		}

		private static final boolean suffixEquals(final long suffixIndex_, final int[] ngram, final int startPos, final int endPos, final HashMap[] localMaps) {
			long suffixIndex = suffixIndex_;
			for (int pos = endPos - 1; pos >= startPos; --pos) {
				HashMap suffixMap = localMaps[pos - startPos];
				final long currKey = suffixMap.getKey(suffixIndex);
				final int firstWord = wordOf(currKey);
				if (ngram[pos] != firstWord) return false;
				if (pos == startPos) return true;
				suffixIndex = prefixOffsetOf(currKey);
			}
			return true;

		}

		private static final boolean implicitSuffixEquals(final long contextOffset_, final int[] ngram, final int startPos, final int endPos,
			final HashMap[] localMaps) {
			long contextOffset = contextOffset_;
			for (int pos = endPos - 1; pos >= startPos; --pos) {
				HashMap suffixMap = localMaps[pos - startPos];
				final long currKey = suffixMap.getKey(contextOffset);
				//				final int firstWord = firstWord(currKey);
				final int firstSearchWord = ngram[pos];
				final long rangeStart = suffixMap.wordRangesLow[firstSearchWord];
				final long rangeEnd = suffixMap.wordRangesHigh[firstSearchWord];
				if (contextOffset < rangeStart || contextOffset >= rangeEnd) return false;
				if (pos == startPos) return true;
				contextOffset = prefixOffsetOf(currKey);
			}
			return true;

		}

		public long getNumHashPositions(int word) {
			if (wordRangesLow == null) return getCapacity();
			return wordRangesHigh[word] - wordRangesLow[word];
		}

		public long getStartOfRange(int word) {
			if (wordRangesLow == null) return 0;
			return wordRangesLow[word];
		}
	}

	private HashMap[] maps;

	private final double maxLoadFactor;

	private final long initialCapacity;

	private long numWords;

	private final boolean useContextEncoding;

	public HashNgramMap(ValueContainer<T> values, NgramMapOpts opts, long initialCapacity) {
		super(values, opts);
		this.cacheSuffixes = opts.cacheSuffixes;
		this.useContextEncoding = opts.storePrefixIndexes;
		this.quadraticProbing = opts.quadraticProbing;
		maps = new HashNgramMap.HashMap[6];
		//		this.defaultVal = defaultVal;
		this.initialCapacity = initialCapacity;
		this.maxLoadFactor = opts.maxLoadFactor;
	}

	/**
	 * For rehashing
	 * 
	 * @param numNgramsForEachWord
	 */
	private HashNgramMap(ValueContainer<T> values, NgramMapOpts opts, long[] capacities, long[][] numNgramsForEachWord) {
		super(values, opts);
		this.cacheSuffixes = opts.cacheSuffixes;
		this.useContextEncoding = opts.storePrefixIndexes;
		this.quadraticProbing = opts.quadraticProbing;
		maps = new HashNgramMap.HashMap[6];
		for (int i = 0; i < capacities.length; ++i) {
			if (capacities[i] < 0) continue;
			maps[i] = new HashMap(capacities[i], numNgramsForEachWord == null ? null : numNgramsForEachWord[i], quadraticProbing);
		}
		//		this.defaultVal = defaultVal;
		this.initialCapacity = -1L;
		this.maxLoadFactor = 0.75;
	}

	@Override
	public long add(int[] ngram, T val) {

		HashMap tightHashMap = maps[ngram.length - 1];
		if (tightHashMap == null) {
			long capacity = ngram.length == 1 ? initialCapacity : maps[ngram.length - 2].getCapacity();
			tightHashMap = maps[ngram.length - 1] = new HashMap(capacity, null, quadraticProbing);
		}
		return addHelp(ngram, 0, ngram.length, val, tightHashMap, true);

	}

	/**
	 * @param ngram
	 * @param val
	 * @param tightHashMap
	 */
	private long addHelp(int[] ngram, int startPos, int endPos, T val, HashMap map, boolean rehashIfNecessary) {

		long key = getKey(ngram, startPos, endPos, true);
		return addHelpWithKey(ngram, startPos, endPos, val, map, key, rehashIfNecessary);
	}

	/**
	 * @param ngram
	 * @param val
	 * @param map
	 * @param hash
	 * @param key
	 * @param rehashIfNecessary
	 * @return
	 */
	private long addHelpWithKey(int[] ngram, int startPos, int endPos, T val, final HashMap map, long key, boolean rehashIfNecessary) {
		long hash = useContextEncoding ? hash(key, ngram[endPos - 1], endPos - startPos - 1, map) : hash(ngram, startPos, endPos, map);
		final long index = map.put(hash, key);
		long prefixIndex = -1;
		if (endPos - startPos > 1) {
			final HashMap prefixMap = maps[endPos - startPos - 2];
			long prefixHash = -1;
			if (useContextEncoding) {
				long prefixKey = getKey(ngram, startPos + 1, endPos, true);
				prefixHash = hash(prefixKey, ngram[endPos - 1], endPos - startPos - 2, prefixMap);
			} else {
				prefixHash = hash(ngram, startPos + 1, endPos, prefixMap);
			}
			prefixIndex = prefixMap.getIndex(ngram, prefixHash, startPos + 1, endPos, maps);
			if (prefixIndex < 0) {
				prefixIndex = addHelp(ngram, startPos, endPos - 1, null, prefixMap, false);
			}
			assert prefixIndex >= 0;
		}
		values.add(endPos - startPos - 1, index, prefixOffsetOf(key), wordOf(key), val, prefixIndex);
		if (rehashIfNecessary && map.getLoadFactor() > maxLoadFactor) {
			rehash(endPos - 1, map.getCapacity() * 3 / 2, false);
		}
		return index;
	}

	private long getKey(int[] ngram, int startPos, int endPos, final boolean addIfNecessary) {
		long key = combineToKey(ngram[startPos], 0);
		if (endPos - startPos == 1) return key;
		for (int ngramOrder = 1; ngramOrder < endPos - startPos; ++ngramOrder) {
			final int currEndPos = startPos + ngramOrder;
			final HashMap currMap = maps[ngramOrder - 1];
			long hash = useContextEncoding ? hash(key, ngram[currEndPos - 1], ngramOrder - 1, currMap) : hash(ngram, startPos, currEndPos, currMap);
			long index = getIndexHelp(ngram, startPos, ngramOrder, currEndPos, hash);
			if (index == -1L) {
				if (addIfNecessary) {
					index = addHelp(ngram, startPos, currEndPos, null, currMap, false);
				} else
					return -1;
			}

			key = combineToKey(ngram[currEndPos], index);
		}
		return key;
	}

	private long hash(long key, int firstWord, int ngramOrder, final HashMap currMap) {
		assert useContextEncoding;
		long hashed = (MurmurHash.hashOneLong(key, 31)) + ngramOrder;
		return processHash(hashed, firstWord, currMap);
	}

	/**
	 * @param ngram
	 * @param startPos
	 * @param ngramOrder
	 * @param currEndPos
	 * @param hash
	 * @return
	 */
	private long getIndexHelp(int[] ngram, int startPos, int ngramOrder, final int endPos, long hash) {
		if (cacheSuffixes) {
			if (cachedLastSuffix[endPos - startPos - 1] != null && equals(ngram, startPos, endPos, cachedLastSuffix[endPos - startPos - 1])) { //
				cachedSuffixHits++;
				return cachedLastIndex[endPos - startPos - 1];
			}
			cachedSuffixMisses++;
		}
		long index = maps[ngramOrder - 1].getIndex(ngram, hash, startPos, endPos, maps);
		if (cacheSuffixes) {
			cachedLastSuffix[endPos - startPos - 1] = getSubArray(ngram, startPos, endPos);
			cachedLastIndex[endPos - startPos - 1] = index;
		}
		return index;
	}

	@Override
	public void handleNgramsFinished(int justFinishedOrder) {
		int ngramOrder = justFinishedOrder - 1;

		if (ngramOrder == 0) numWords = maps[ngramOrder].numFilled;

	}

	@Override
	public long getOffset(int[] ngram, int startPos, int endPos) {
		if (containsOutOfVocab(ngram, startPos, endPos)) return -1;

		//		hashTimer.start();
		final HashMap tightHashMap = maps[endPos - startPos - 1];
		long hash = hash(ngram, startPos, endPos, tightHashMap);
		if (hash < 0) return -1;
		//		hashTimer.accumStop();
		//		indexTimer.start();
		final long index = tightHashMap.getIndex(ngram, hash, startPos, endPos, maps);
		//		indexTimer.accumStop();
		return index;
	}

	@Override
	public void trim() {

		if (opts.storeWordsImplicitly) {
			rehash(-1, -1, true);
		}
		for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
			if (maps[ngramOrder] == null) break;
			values.trimAfterNgram(ngramOrder, maps[ngramOrder].getCapacity());
			//			Logger.logss("Load factor for " + (ngramOrder + 1) + ": " + maps[ngramOrder].getLoadFactor());

		}
		//		Logger.logss("Average number of linear probes during building: " + (double) numProbes / numQueries);
		numProbes = numQueries = 0;

	}

	@Override
	public boolean getValuesDirectly() {
		return false;
	}

	@Override
	public T getValue(int[] ngram, int startPos, int endPos, LmContextInfo prefixIndex) {
		long index = getOffset(ngram, startPos, endPos);
		return values.getFromOffset(index, endPos - startPos);
	}

	/**
	 * @param ngram
	 * @param endPos
	 * @param startPos
	 * @return
	 */
	private long hash(int[] ngram, int startPos, int endPos, final HashMap currMap) {
		if (useContextEncoding) {
			long key = getKey(ngram, startPos, endPos, false);
			if (key < 0) return -1;
			return hash(key, ngram[endPos - 1], endPos - startPos - 1, currMap);
		}
		int l = MurmurHash.hash32(ngram, startPos, endPos, PRIME);
		if (l < 0) l = -l;
		final int firstWord = ngram[endPos - 1];
		return processHash(l, firstWord, currMap);
	}

	/**
	 * @param startPos
	 * @param endPos
	 * @param hash
	 * @param firstWord
	 * @return
	 */
	private long processHash(final long hash_, final int firstWord, final HashMap currMap) {
		long hash = hash_;
		if (hash < 0) hash = -hash;
		hash = (int) (hash % currMap.getNumHashPositions(firstWord));
		return hash + currMap.getStartOfRange(firstWord);
	}

	private static long mask(int i, int bitOffset) {
		return ((1L << i) - 1L) << bitOffset;
	}

	private static long prefixOffsetOf(long currKey) {
		return (currKey & SUFFIX_MASK) >>> INDEX_OFFSET;
	}

	private static int wordOf(long currKey) {
		return (int) (currKey >>> WORD_BIT_OFFSET);
	}

	private static long combineToKey(int word, long suffix) {
		return ((long) word << WORD_BIT_OFFSET) | (suffix << INDEX_OFFSET);
	}

	private void rehash(int changedNgramOrder, long newCapacity, boolean storeWordsImplicitly) {
		ValueContainer<T> newValues = values.createFreshValues();
		long[] newCapacities = new long[maps.length];
		Arrays.fill(newCapacities, -1L);
		long[][] numNgramsForEachWord = null;
		if (storeWordsImplicitly) {
			numNgramsForEachWord = new long[maps.length][(int) numWords];
			for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
				final HashMap currMap = maps[ngramOrder];
				if (currMap == null) continue;
				for (long actualIndex = 0; actualIndex < currMap.getCapacity(); ++actualIndex) {
					long key = currMap.getKey(actualIndex);
					if (key == HashMap.EMPTY_KEY) continue;
					int[] ngram = getNgram(key, ngramOrder);
					final int firstWordOfNgram = ngram[ngramOrder];
					numNgramsForEachWord[ngramOrder][firstWordOfNgram]++;
				}

				for (int i = 0; i < numNgramsForEachWord[ngramOrder].length; ++i) {
					final long numNgrams = numNgramsForEachWord[ngramOrder][i];
					numNgramsForEachWord[ngramOrder][i] = numNgrams <= 3 ? numNgrams : Math.round(numNgrams * 1.0 / maxLoadFactor);
				}
			}
			for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
				if (maps[ngramOrder] == null) break;
				newCapacities[ngramOrder] = sum(numNgramsForEachWord[ngramOrder]);
			}
		} else {

			for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
				if (maps[ngramOrder] == null) break;
				newCapacities[ngramOrder] = ngramOrder == changedNgramOrder ? newCapacity : maps[ngramOrder].getCapacity();
			}
		}

		HashNgramMap<T> newMap = new HashNgramMap<T>(newValues, opts, newCapacities, numNgramsForEachWord);

		for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
			final HashMap currMap = maps[ngramOrder];
			if (currMap == null) continue;
			for (long actualIndex = 0; actualIndex < currMap.getCapacity(); ++actualIndex) {
				long key = currMap.getKey(actualIndex);
				if (key == HashMap.EMPTY_KEY) continue;
				int[] ngram = getNgram(key, ngramOrder);

				//				final long oldIndex = hashCodes[ngramOrder].getIndex(key);
				final T val = values.getFromOffset(actualIndex, ngramOrder);
				newMap.addHelp(ngram, 0, ngram.length, val, newMap.maps[ngramOrder], false);

			}
		}
		maps = newMap.maps;

		//		if (changedNgramOrder < 0) {
		//			// rehashing to store words implicitly, need to trim
		//			for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
		//				final CoalescedHashMap currMap = maps[ngramOrder];
		//				if (currMap == null) continue;
		//				newValues.trimAfterNgram(ngramOrder, currMap.getCapacity());
		//			}
		//		}
		values.setFromOtherValues(newValues);

	}

	private static long sum(long[] array) {
		long sum = 0;
		for (long l : array)
			sum += l;
		return sum;
	}

	private int[] getNgram(final long key_, final int ngramOrder_) {
		long key = key_;
		int ngramOrder = ngramOrder_;
		int[] l = new int[ngramOrder + 1];
		int firstWord = wordOf(key);
		int k = l.length - 1;
		l[k] = firstWord;
		k--;
		while (ngramOrder > 0) {
			long suffixIndex = prefixOffsetOf(key);
			key = maps[ngramOrder - 1].getKey(suffixIndex);
			ngramOrder--;
			firstWord = wordOf(key);
			l[k] = firstWord;
			k--;
		}
		return l;
	}

	@Override
	public void logTimingInfo() {
		//		Logger.logss("Average number of linear probes during querying: " + (double) numQueryProbes / numQueryCalls);
	}

	@Override
	public void initWithLengths(List<Long> numNGrams) {
		maps = new HashMap[numNGrams.size()];
		for (int i = 0; i < numNGrams.size(); ++i) {
			final long l = numNGrams.get(i);
			final long size = Math.round(l / maxLoadFactor) + 1;
			maps[i] = new HashMap(size, null, quadraticProbing);
			values.setSizeAtLeast(size, i);

		}
	}

	@Override
	public long getOffset(long suffixIndex_, int suffixNgramOrder, int firstWord) {
		final long suffixIndex = suffixNgramOrder < 0 ? 0 : suffixIndex_;
		assert suffixIndex >= 0;
		if (opts.storeWordsImplicitly) throw new RuntimeException("Not yet implemented");
		int ngramOrder = suffixNgramOrder + 1;

		//		int[] ngram = getNgram(combineToKey(firstWord, suffixIndex), ngramOrder);
		//		hashTimer.start();
		//		int startPos = 0;
		//		int endPos = ngram.length;
		final long key = combineToKey(firstWord, suffixIndex);
		final HashMap tightHashMap = maps[ngramOrder];
		final long hash = hash(key, firstWord, ngramOrder, tightHashMap);
		//		hashTimer.accumStop();
		//		indexTimer.start();
		final long index = tightHashMap.getIndex(key, suffixIndex, firstWord, hash);
		//		indexTimer.accumStop();
		return index;
	}

}
