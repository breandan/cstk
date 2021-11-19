package edu.berkeley.nlp.langmodel.impl;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.util.StringIndexer;

/**
 * Default implementation of all NGramLanguageModel functionality except the
 * getLogProb(int[] ngram, int startPos, int endPos) function.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public abstract class AbstractNgramLanguageModel implements EfficientNGramLanguageModel, Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected final int lmOrder;

	protected final StringIndexer wordIndexer;

	public AbstractNgramLanguageModel(int lmOrder, StringIndexer wordIndexer) {
		this.lmOrder = lmOrder;
		this.wordIndexer = wordIndexer;
	}

	public int getLmOrder() {
		return lmOrder;
	}

	public float scoreSequence(List<String> sequence) {
		return EfficientNGramLanguageModel.DefaultImplementations.scoreSequence(sequence, this);
	}

	public float scoreSentence(List<String> sentence) {
		return EfficientNGramLanguageModel.DefaultImplementations.scoreSentence(sentence, this);
	}


	public float getLogProb(List<String> phrase) {
		return EfficientNGramLanguageModel.DefaultImplementations.getLogProb(phrase, this);
	}

	public float getLogProb(int[] ngram) {
		return EfficientNGramLanguageModel.DefaultImplementations.getLogProb(ngram, this);
	}

	public StringIndexer getWordIndexer() {
		return wordIndexer;
	}

}
