package edu.berkeley.nlp.langmodel.impl;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.util.StringIndexer;

/**
 * 
 * Default implementation of all ContextEncodedNgramLanguageModel functionality
 * except <code> 
 *   getLogProb(long context, int contextOrder, int word, LmStateOutput outputContext) </code>
 * and
 * <code>getLogProb(int[] ngram, int startPos, int endPos, LmStateOutput outputContext) </code>
 * 
 * @author adampauls
 * 
 * @param <String>
 */
public abstract class AbstractContextEncodedNgramLanguageModel extends AbstractNgramLanguageModel implements ContextEncodedNgramLanguageModel, Serializable
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public AbstractContextEncodedNgramLanguageModel(int lmOrder, StringIndexer wordIndexer) {
		super(lmOrder, wordIndexer);
	}

	@Override
	public int getLmOrder() {
		return lmOrder;
	}

	public float getLogProb(List<String> phrase, LmContextInfo contextOutput) {
		return ContextEncodedNgramLanguageModel.DefaultImplementations.getLogProb(phrase, contextOutput, this);
	}

	public float getLogProb(int[] ngram, int startPos, int endPos) {
		return ContextEncodedNgramLanguageModel.DefaultImplementations.getLogProb(ngram, startPos, endPos, this);
	}

}
