package edu.berkeley.nlp.langmodel;

import java.util.List;

public interface LanguageModelFactory
{
	/**
	 * Constructs a language model implementation.
	 * 
	 * @param trainingData
	 *            An Iterable over English sentences represented as Lists of
	 *            Strings.
	 * @return
	 */
	public NgramLanguageModel newLanguageModel(Iterable<List<String>> trainingData);

}
