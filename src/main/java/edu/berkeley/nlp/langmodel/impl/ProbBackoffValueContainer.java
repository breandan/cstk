package edu.berkeley.nlp.langmodel.impl;

import edu.berkeley.nlp.util.Indexer;

public final class ProbBackoffValueContainer extends LmValueContainer<ProbBackoffPair>
{

	private static final long serialVersionUID = 964277160049236607L;

	float[][] probsForRank;

	float[][] backoffsForRank;

	public ProbBackoffValueContainer(int valueRadix, int countCutoffForHuffman, boolean useHuffman, boolean storePrefixes) {
		this(new Indexer<ProbBackoffPair>(), valueRadix, countCutoffForHuffman, useHuffman, storePrefixes);
	}

	private ProbBackoffValueContainer(Indexer<ProbBackoffPair> countIndexer, int valueRadix, int countCutoffForHuffman, boolean useHuffman, boolean storePrefixes) {
		super(countIndexer, valueRadix, countCutoffForHuffman, useHuffman, storePrefixes);
	}

	public ProbBackoffValueContainer createFreshValues() {
		return new ProbBackoffValueContainer(countIndexer, valueRadix, countCutoffForHuffman, useHuffman, storePrefixIndexes);
	}

	public final float getProb(int ngramOrder, long index) {
		return getCount(ngramOrder, index, probsForRank[ngramOrder]);
	}

	/**
	 * @param ngramOrder
	 * @param index
	 * @param uncompressProbs2
	 * @return
	 */
	private float getCount(int ngramOrder, long index, final float[] array) {
		int countIndex = valueRanksCompressed[ngramOrder][(int) index];
		return array[countIndex];
	}

	public final float getBackoff(int ngramOrder, long index) {
		return getCount(ngramOrder, index, backoffsForRank[ngramOrder]);
	}

	@Override
	protected ProbBackoffPair getDefaultVal() {
		return new ProbBackoffPair(Float.NaN, Float.NaN);
	}

	@Override
	protected void storeCounts(Indexer<ProbBackoffPair> countIndexer_, int[] indexToSortedIndexMap, int ngramOrder) {
		if (probsForRank == null) probsForRank = new float[6][];
		if (backoffsForRank == null) backoffsForRank = new float[6][];
		probsForRank[ngramOrder] = new float[countIndexer_.size()];
		backoffsForRank[ngramOrder] = new float[countIndexer_.size()];
		int k = 0;
		for (ProbBackoffPair pair : countIndexer) {
			final int i = indexToSortedIndexMap[k];
			k++;
			if (i < 0) continue;
			probsForRank[ngramOrder][i] = pair.prob;
			backoffsForRank[ngramOrder][i] = pair.backoff;
		}
	}

	@Override
	protected ProbBackoffPair getCount(int index, int ngramOrder) {
		return new ProbBackoffPair(probsForRank[ngramOrder][index], backoffsForRank[ngramOrder][index]);
	}

}