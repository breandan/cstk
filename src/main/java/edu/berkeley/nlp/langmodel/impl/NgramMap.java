package edu.berkeley.nlp.langmodel.impl;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.langmodel.impl.ContextEncodedNgramLanguageModel.LmContextInfo;

public abstract class NgramMap<T> implements Serializable
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected final ValueContainer<T> values;

	protected final NgramMapOpts opts;

	protected NgramMap(ValueContainer<T> values, NgramMapOpts opts) {
		this.values = values;
		this.opts = opts;
	}

	protected static boolean equals(int[] ngram, int startPos, int endPos, int[] cachedNgram) {
		if (cachedNgram.length != endPos - startPos) return false;
		for (int i = 0; i < endPos - startPos; ++i) {
			if (ngram[startPos + i] != cachedNgram[i]) return false;
		}
		return true;
	}

	protected static int[] getSubArray(int[] ngram, int startPos, int endPos) {
		return Arrays.copyOfRange(ngram, startPos, endPos);

	}

	protected static boolean containsOutOfVocab(final int[] ngram, int startPos, int endPos) {
		for (int i = startPos; i < endPos; ++i) {
			if (ngram[i] < 0) return true;
		}
		return false;
	}

	public abstract long add(int[] ngram, T val);

	public abstract void handleNgramsFinished(int justFinishedOrder);

	public abstract long getOffset(int[] phrase, int startPos, int endPos);

	public abstract void trim();

	abstract public boolean getValuesDirectly();

	abstract public T getValue(int[] ngram, int startPos, int endPos, LmContextInfo contextOutput);

	public void logTimingInfo() {

	}

	abstract public void initWithLengths(List<Long> numNGrams);

	abstract public long getOffset(long contextOffset, int prefixNgramOrder, int word);

	public ValueContainer<T> getValues() {
		return values;
	}

}