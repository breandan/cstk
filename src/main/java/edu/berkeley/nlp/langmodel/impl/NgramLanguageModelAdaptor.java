package edu.berkeley.nlp.langmodel.impl;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;

/**
 * Wraps an implementation of the context-encoded n-gram language model
 * described in
 * http://www.cs.berkeley.edu/~klein/cs288/sp11/restricted/faster-smaller.pdf.
 * 
 * You can retrieve an object which implements the
 * ContextEncodedNgramLanguageModel interface by galling getContextEncodedLm()
 * 
 * @author adampauls
 * 
 */
public class NgramLanguageModelAdaptor implements NgramLanguageModel
{

	private final ContextEncodedNgramLanguageModel lm;

	public ContextEncodedNgramLanguageModel getContextEncodedLm() {
		return lm;
	}

	public NgramLanguageModelAdaptor(ContextEncodedNgramLanguageModel lm) {
		this.lm = lm;
	}

	public int getOrder() {
		return lm.getLmOrder();
	}

	public double getNgramLogProbability(int[] ngram, int from, int to) {
		return lm.getLogProb(ngram, from, to);
	}

	public long getCount(int[] ngram) {
		throw new UnsupportedOperationException("Method not implemented");
	}

}
