package edu.berkeley.nlp.langmodel;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.util.CollectionUtils;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.
 * 
 * @author Dan Klein
 */
public class EmpiricalUnigramLanguageModel implements NgramLanguageModel
{

	public static class EmpiricalUnigramLanguageModelFactory implements LanguageModelFactory
	{

		public NgramLanguageModel newLanguageModel(Iterable<List<String>> trainingData) {
			return new EmpiricalUnigramLanguageModel(trainingData);
		}
	}

	static final String STOP = NgramLanguageModel.STOP;

	long total = 0;

	long[] wordCounter = new long[10];

	public EmpiricalUnigramLanguageModel(Iterable<List<String>> sentenceCollection) {
		System.out.println("Building EmpiricalUnigramLanguageModel . . .");
		int sent = 0;
		for (List<String> sentence : sentenceCollection) {
			sent++;
			if (sent % 1000000 == 0) System.out.println("On sentence " + sent);
			List<String> stoppedSentence = new ArrayList<String>(sentence);
			stoppedSentence.add(0, NgramLanguageModel.START);
			stoppedSentence.add(STOP);
			for (String word : stoppedSentence) {
				int index = EnglishWordIndexer.getIndexer().addAndGetIndex(word);
				if (index >= wordCounter.length) wordCounter = CollectionUtils.copyOf(wordCounter, wordCounter.length * 2);
				wordCounter[index]++;
			}
		}
		System.out.println("Done building EmpiricalUnigramLanguageModel.");
		wordCounter = CollectionUtils.copyOf(wordCounter, EnglishWordIndexer.getIndexer().size());
		total = CollectionUtils.sum(wordCounter);
	}



	public int getOrder() {
		return 1;
	}

	public double getNgramLogProbability(int[] ngram, int from, int to) {
		double logProb = 0.0;
		for (int i = from; i < to; ++i) {

			final int word = ngram[i];
			double count = (word < 0 || word >= wordCounter.length) ? 1.0 : wordCounter[word];

			logProb += Math.log(count / (total + 1.0));
		}
		return logProb;
	}

	public long getCount(int[] ngram) {
		if (ngram.length > 1) return 0;
		final int word = ngram[0];
		if (word < 0 || word >= wordCounter.length) return 0;
		return wordCounter[word];
	}
}
