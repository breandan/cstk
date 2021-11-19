package edu.berkeley.nlp.langmodel.impl;

import java.io.Serializable;

import edu.berkeley.nlp.util.StringIndexer;

/**
 * Language model implementation which uses Katz-style backoff computation.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class KneserNeyLm extends AbstractContextEncodedNgramLanguageModel implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected final NgramMap<ProbBackoffPair> map;

	private final ProbBackoffValueContainer values;

	/**
	 * Fixed constant returned when computing the log probability for an n-gram
	 * whose last word is not in the vocabulary. Note that this is different
	 * from the log prob of the <code>unk</code> tag probability.
	 * 
	 */
	private final float oovWordLogProb;

	public static KneserNeyLm fromFile(NgramMapOpts opts, String lmFile, int lmOrder, StringIndexer wordIndexer) {
		return fromFile(opts, lmFile, lmOrder, wordIndexer, true);
	}

	public static KneserNeyLm fromFile(NgramMapOpts opts, String lmFile, int lmOrder, StringIndexer wordIndexer, boolean lockIndexer) {
		ProbBackoffValueContainer values = new ProbBackoffValueContainer(opts.valueRadix, opts.huffmanCountCutoff, opts.useHuffman, opts.storePrefixIndexes);
		NgramMap<ProbBackoffPair> map = new HashNgramMap<ProbBackoffPair>(values, opts, 100000);

		new ARPALmReader(lmFile, wordIndexer, lmOrder).parse(new NgramMapAddingCallback<ProbBackoffPair>(map));
		//		if (lockIndexer) wordIndexer.

		return new KneserNeyLm(lmOrder, wordIndexer, map, opts);
	}

	public KneserNeyLm(int lmOrder, StringIndexer wordIndexer, NgramMap<ProbBackoffPair> map, NgramMapOpts opts) {
		super(lmOrder, wordIndexer);
		oovWordLogProb = opts.unknownWordLogProb;
		this.map = map;
		this.values = (ProbBackoffValueContainer) map.getValues();

	}

	public float getLogProb(int[] ngram, int startPos, int endPos, LmContextInfo prefixIndex) {
		final float score = scoreLog10HelpWithCache(ngram, startPos, endPos, prefixIndex, true);
		final float log10ln = score;//convertLog ? convertFromLogBase10(score) : score;
		return log10ln;
	}

	public float getLogProb(long context, int contextOrder, int word, LmContextInfo outputPrefixIndex) {
		final float score = scoreLog10HelpWithCache(context, contextOrder, word, outputPrefixIndex);
		final float log10ln = score;//convertLog ? convertFromLogBase10(score) : score;
		return log10ln;
	}

	private float scoreLog10HelpWithCache(final int[] ngram, final int startPos, final int endPos, LmContextInfo prefixIndex, boolean topLevel) {
		return scoreLog10Help(ngram, startPos, endPos, prefixIndex, topLevel);
	}

	private float scoreLog10Help(final int[] ngram, final int startPos, final int endPos, LmContextInfo prefixIndex, boolean topLevel) {
		final NgramMap<ProbBackoffPair> localMap = map;
		if (localMap.getValuesDirectly()) {
			ProbBackoffPair pair = localMap.getValue(ngram, startPos, endPos, prefixIndex);
			if (pair != null) {
				return pair.prob;
			} else {
				if (endPos - startPos > 1) {
					final float backoffProb = scoreLog10HelpWithCache(ngram, startPos + 1, endPos, prefixIndex, false);
					ProbBackoffPair backoffPair = localMap.getValue(ngram, startPos, endPos - 1, null);
					float backOff = backoffPair == null ? 0.0f : backoffPair.backoff;
					return backOff + backoffProb;
				} else {
					return oovWordLogProb;
				}
			}
		} else {
			long index = localMap.getOffset(ngram, startPos, endPos);
			if (index >= 0) {
				final int ngramOrder = endPos - startPos - 1;
				final float prob = values.getProb(ngramOrder, index);
				if (prefixIndex != null) {
					if (topLevel) {
						long prefixIndexHere = values.getContextOffset(index, ngramOrder);
						assert prefixIndexHere >= 0;

						prefixIndex.context = prefixIndexHere;
						prefixIndex.order = (ngramOrder - 1);
					} else {
						prefixIndex.context = index;
						prefixIndex.order = ngramOrder;
					}
				}
				return prob;
			} else {
				if (endPos - startPos > 1) {
					final float backoffProb = scoreLog10HelpWithCache(ngram, startPos + 1, endPos, prefixIndex, false);
					long backoffIndex = localMap.getOffset(ngram, startPos, endPos - 1);
					float backOff = backoffIndex < 0 ? 0.0f : values.getBackoff(endPos - startPos - 2, backoffIndex);
					return backOff + backoffProb;
				} else {
					return oovWordLogProb;
				}
			}
		}
	}

	private float scoreLog10HelpWithCache(long inputPrefix, int prefixNgramOrder, int word, LmContextInfo outputPrefixIndex) {
		return scoreLog10Help(inputPrefix, prefixNgramOrder, word, outputPrefixIndex);
	}

	private float scoreLog10Help(long inputPrefix, int prefixNgramOrder, int word, LmContextInfo outputPrefixIndex) {

		final NgramMap<ProbBackoffPair> localMap = map;
		if (localMap.getValuesDirectly()) throw new RuntimeException("Compressed version not set up for index querying yet");

		long index = localMap.getOffset(inputPrefix, prefixNgramOrder, word);
		if (index >= 0) {
			final int ngramOrder = prefixNgramOrder + 1;
			final float prob = values.getProb(ngramOrder, index);
			long prefixIndexHere = values.getContextOffset(index, ngramOrder);
			if (outputPrefixIndex != null) {
				outputPrefixIndex.context = prefixIndexHere;
				outputPrefixIndex.order = ngramOrder;
			}
			return prob;
		} else if (prefixNgramOrder >= 0) {
			final int nextPrefixOrder = prefixNgramOrder - 1;
			long nextPrefixIndex = nextPrefixOrder < 0 ? 0 : values.getContextOffset(inputPrefix, prefixNgramOrder);
			final float nextProb = scoreLog10HelpWithCache(nextPrefixIndex, nextPrefixOrder, word, outputPrefixIndex);
			long backoffIndex = inputPrefix;
			float backOff = backoffIndex < 0 ? 0.0f : values.getBackoff(prefixNgramOrder, backoffIndex);
			return backOff + nextProb;
		} else {
			return oovWordLogProb;
		}

	}

}
