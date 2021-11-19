package edu.berkeley.nlp.langmodel;

import java.util.List;


/**
 * A dummy trigram language model which always returns 0.
 * 
 * @author Dan Klein
 */
public class StubLanguageModel implements NgramLanguageModel
{

	public static class StubLanguageModelFactory implements LanguageModelFactory
	{

		public NgramLanguageModel newLanguageModel(Iterable<List<String>> trainingData) {
			return new StubLanguageModel();
		}
	}

	public StubLanguageModel() {

	}

	public int getOrder() {
		return 3;
	}

	public double getNgramLogProbability(int[] ngram, int from, int to) {
		return 0.0;
	}

	public long getCount(int[] ngram) {
		return 0;
	}
}
