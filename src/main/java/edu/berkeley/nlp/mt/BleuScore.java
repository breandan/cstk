package edu.berkeley.nlp.mt;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.StrUtils;

public class BleuScore implements Serializable {
	private static final long serialVersionUID = 1L;
	private double[] matches, guesses, precisions;
	private double candidateLength;
	private double referenceLength;
	private int max;
	private double lengthPenalty, bleu, logBleu;

	private double smoothConstant = 0.0;
	
	public BleuScore(NgramMultiset c, NgramMultiset r, int max, double smoothConstant)
	{
		this(max, smoothConstant);
		setCandidateLength(c.getLength());
		referenceLength = r.getLength();

		// Compute precision statistics
		for (int i = 0; i < max; i++) {
			Counter<String> ccounts = c.getNgrams(i);
			Counter<String> rcounts = r.getNgrams(i);
			double imatches = 0.0;
			for (String ngram : ccounts.keySet()) {
				imatches += Math.min(ccounts.getCount(ngram), rcounts.getCount(ngram));
			}
			matches[i] = imatches;
			guesses[i] = ccounts.totalCount();
		}
		computeBleu();
	}

	/**
	 * Scores a corpus from a list of sub-corpora (usually sentences)
	 * 
	 * @param scores
	 */
	public BleuScore(List<BleuScore> scores) {
		assert (scores.size() >= 1);
		this.max = scores.get(0).max;
		tallyScores(scores);
		computeBleu();
	}

	private BleuScore(int max, double smoothConstant)
	{
		this.max = max;
		this.smoothConstant = smoothConstant;
		matches = new double[max];
		guesses = new double[max];
	}

	public BleuScore(NgramMultiset c, NgramMultiset r, double smoothConstant)
	{
		this(c, r, 4, smoothConstant);
	}

	public BleuScore(String cand, String ref) {
		this(new ReferenceSet(cand), new ReferenceSet(ref), 0.0);
	}

	public BleuScore(List<String> candWords, List<String> refWords)
	{
		this(new ReferenceSet(StrUtils.join(candWords)), new ReferenceSet(StrUtils.join(refWords)), 0.0);
	}

	public BleuScore(List<String> candWords, NgramMultiset ref)
	{
		this(new ReferenceSet(StrUtils.join(candWords)), ref, 0.0);
	}

	public BleuScore(List<String> candWords, NgramMultiset ref, double smoothConstant)
	{
		this(new ReferenceSet(StrUtils.join(candWords)), ref, smoothConstant);
	}

	public BleuScore(List<BleuScore> scores, BleuScore oneMore) {
		this.max = oneMore.max;
		tallyScores(scores);
		for (int i = 0; i < max; i++) {
			matches[i] += oneMore.matches[i];
			guesses[i] += oneMore.guesses[i];
		}
		setCandidateLength(getCandidateLength() + oneMore.getCandidateLength());
		referenceLength += oneMore.referenceLength;
		computeBleu();
	}

	public BleuScore(NgramMultiset cand, NgramMultiset ref)
	{
		this(cand, ref, 4, 0.0);
	}

	private void tallyScores(List<BleuScore> scores) {
		matches = new double[max];
		guesses = new double[max];
		for (BleuScore score : scores) {
			for (int i = 0; i < max; i++) {
				matches[i] += score.matches[i];
				guesses[i] += score.guesses[i];
			}
			setCandidateLength(getCandidateLength() + score.getCandidateLength());
			referenceLength += score.referenceLength;
		}
	}

	private void computeBleu() {
		double meanPrecision = 1;
		precisions = new double[max];
		for (int i = 0; i < max; i++) {
			precisions[i] = (matches[i] + smoothConstant) / (guesses[i] + smoothConstant);
			meanPrecision *= precisions[i];
		}

		// Compute bleu
		double lengthRatio = Math.max(1.0, 1.0 * referenceLength / getCandidateLength());
		lengthPenalty = Math.exp(1 - lengthRatio);
		bleu = lengthPenalty * Math.pow(meanPrecision, 1.0 / max);
		logBleu = Math.log(bleu);
	}

	public double[] getMatches() {
		return matches;
	}

	public double[] getGuesses() {
		return guesses;
	}

	public double[] getPrecisions() {
		return precisions;
	}

	public double getLengthPenalty() {
		return lengthPenalty;
	}

	public double getBleu() {
		return bleu;
	}

	public double getLogBleu() {
		return logBleu;
	}

	/* Static scoring */

	public static Counter<String> countNgrams(int oneLessThanLen, List<String> words) {
		Counter<String> lengrams = new Counter<String>();
		for (int i = 0; i < words.size() - oneLessThanLen; i++) {
			int j = i + oneLessThanLen + 1;
			String ngram = StrUtils.join(words.subList(i, j));
			lengrams.incrementCount(ngram, 1);
		}
		assert (lengrams.totalCount() == Math.max(0, words.size() - oneLessThanLen));
		return lengrams;
	}

	/**
	 * Creates a Bleu Score with dummy counts
	 */
	public static BleuScore createDummyBleuScore(double smoothNum, double smoothDenom,
			int max) {
		BleuScore bs = new BleuScore(max, 0.0);
		for (int i = 0; i < max; i++) {
			bs.matches[i] = smoothNum;
			bs.guesses[i] = smoothDenom;
		}
		return bs;
	}

	public static BleuScore createDummyBleuScore(double smoothingAddend) {
		return createDummyBleuScore(smoothingAddend, smoothingAddend, 4);
	}

	public static BleuScore createDummyBleuScore(double smoothingNumerator,
			double smoothingDenominator) {
		return createDummyBleuScore(smoothingNumerator, smoothingDenominator, 4);
	}

	@Override
	public String toString() {
		return "BLEU(" + formatDouble(100 * getBleu()) + ")";
	}

	public static String formatDouble(double x) {
		if (Math.abs(x - (int) x) < 1e-40) // An integer (probably)
			return "" + (int) x;
		if (Math.abs(x) < 1e-3) // Scientific notation (close to 0)
			return String.format("%.2e", x);
		return String.format("%.3f", x);
	}

	public String fullString() {
		return String.format("%.2f (%.1f, %.1f, %.1f, %.1f; %.2f len. pen.)",
				getBleu() * 100, precisions[0] * 100, precisions[1] * 100,
				precisions[2] * 100, precisions[3] * 100, lengthPenalty * 100);
	}



	public static BleuScore bleuFromVectors(double[][] hypVectors, double[][] refVectors) {
		BleuScore bs = new BleuScore(4, 0.0);
		double refLength = 0;
		for (int n = 0; n < 4; n++) {
			double matches = 0, guesses = 0;
			double[] hypCounts = hypVectors[n];
			double[] refCounts = refVectors[n];
			int len = Math.max(hypCounts.length, refCounts.length);
			for (int g = 0; g < len; g++) {
				if (n == 0 && g < refCounts.length) refLength += refCounts[g];
				if (g < hypCounts.length) {
					guesses += hypCounts[g];
					if (g < refCounts.length) {
						matches += Math.min(hypCounts[g], refCounts[g]);
					}
				}
			}
			bs.matches[n] = matches;
			bs.guesses[n] = guesses;
		}
		bs.referenceLength = refLength;
		bs.setCandidateLength(bs.guesses[0]);
		bs.computeBleu();
		return bs;
	}

	public void setCandidateLength(double candidateLength) {
		this.candidateLength = candidateLength;
	}

	public double getCandidateLength() {
		return candidateLength;
	}

	public static BleuScore scoreFromWords(List<String> candidate,
			List<String> reference) {
		ReferenceSet cand = new ReferenceSet(candidate);
		ReferenceSet ref = new ReferenceSet(reference);
		return new BleuScore(cand, ref, 0.0);
	}

}
