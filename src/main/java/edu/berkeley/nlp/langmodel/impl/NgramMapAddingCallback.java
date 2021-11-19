package edu.berkeley.nlp.langmodel.impl;

import java.util.List;

import edu.berkeley.nlp.mt.decoder.Logger;

/**
 * Reader callback which adds n-grams to an NgramMap
 * 
 * @author adampauls
 * 
 * @param <V>
 *            Value type
 */
public final class NgramMapAddingCallback<V> implements LmReaderCallback<V>
{
	private final NgramMap<V> map;

	int warnCount = 0;

	public NgramMapAddingCallback(NgramMap<V> map) {
		this.map = map;
	}

	public void call(int[] ngram, V v, String words) {
		long add = map.add(ngram, v);
		if (add < 0) {
			if (warnCount >= 0 && warnCount < 10) {
				Logger.warn("Could not add line " + words + "\nThis is usually because the prefix for the n-grams was not already in the map");
				warnCount++;
			}
			if (warnCount > 10) warnCount = -1;
		}
	}

	public void handleNgramOrderFinished(int order) {
		map.handleNgramsFinished(order);
	}

	public void cleanup() {
		map.trim();
	}

	public void initWithLengths(List<Long> numNGrams) {
		map.initWithLengths(numNGrams);
	}
}