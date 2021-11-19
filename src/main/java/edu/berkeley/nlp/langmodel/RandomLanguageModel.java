package edu.berkeley.nlp.langmodel;

import java.util.Random;

import edu.berkeley.nlp.mt.decoder.MurmurHash;

public class RandomLanguageModel implements NgramLanguageModel
{

	public int getOrder() {
		return 3;
	}

	public double getNgramLogProbability(int[] ngram, int from, int to) {
		int hash32 = MurmurHash.hash32(ngram, from, to, 31);
		double prob = Math.log(new Random(hash32).nextDouble());
		return prob;

	}

	public long getCount(int[] ngram) {
		throw new UnsupportedOperationException("Method not yet implemented");
	}

}
