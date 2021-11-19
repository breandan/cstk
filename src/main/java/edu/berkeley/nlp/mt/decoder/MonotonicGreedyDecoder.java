package edu.berkeley.nlp.mt.decoder;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.PhraseTableForSentence;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;
import edu.berkeley.nlp.util.CollectionUtils;

/**
 * A very simple (and bad) monotonic decoder implementation. This implementation
 * greedily adds the best (highest-scoring) translation of an uncovered foreign
 * span that is yet to be consumed until the whole foreign sentence is consumed.
 */
public class MonotonicGreedyDecoder implements Decoder
{
	private PhraseTable tm;

	private NgramLanguageModel lm;

	private DistortionModel dm;

	/**
	 * @param tm
	 * @param lm
	 */
	public MonotonicGreedyDecoder(PhraseTable tm, NgramLanguageModel lm, DistortionModel dm) {
		super();
		this.tm = tm;
		this.lm = lm;
		this.dm = dm;
	}

	public static class MonotonicGreedyDecoderFactory implements DecoderFactory
	{

		public Decoder newDecoder(PhraseTable tm, NgramLanguageModel lm, DistortionModel dm) {
			return new MonotonicGreedyDecoder(tm, lm, dm);
		}

	}

	public List<ScoredPhrasePairForSentence> decode(List<String> sentence) {
		int length = sentence.size();
		PhraseTableForSentence tmState = tm.initialize(sentence);
		int start = 0;
		List<ScoredPhrasePairForSentence> ret = new ArrayList<ScoredPhrasePairForSentence>();
		int[] lmContextBuf = new int[lm.getOrder() + tmState.getMaxPhraseLength() + 1];
		lmContextBuf[0] = EnglishWordIndexer.getIndexer().addAndGetIndex(NgramLanguageModel.START);
		int currLmContextLength = 1;
		double totalScore = 0.0;
		while (start < length) {
			ScoredPhrasePairForSentence best = null;
			int bestLmContextLength = -1;
			int[] bestLmContextBuf = null;

			double max = Double.NEGATIVE_INFINITY;
			if (currLmContextLength >= lm.getOrder()) {
				System.arraycopy(lmContextBuf, currLmContextLength - lm.getOrder() + 1, lmContextBuf, 0, lm.getOrder() - 1);
				currLmContextLength = lm.getOrder() - 1;
			}
			for (int end = start + 1; end <= start + tmState.getMaxPhraseLength(); ++end) {
				List<ScoredPhrasePairForSentence> scoreSortedTranslationsForSpan = tmState.getScoreSortedTranslationsForSpan(start, end);
				if (scoreSortedTranslationsForSpan != null) {
					for (final ScoredPhrasePairForSentence translation : scoreSortedTranslationsForSpan) {
						double score = translation.score;
						System.arraycopy(translation.english.indexedEnglish, 0, lmContextBuf, currLmContextLength,
							translation.english.indexedEnglish.length);
						int currTrgLength = currLmContextLength + translation.english.indexedEnglish.length;
						if (end == length) {
							lmContextBuf[currTrgLength] = EnglishWordIndexer.getIndexer().addAndGetIndex(NgramLanguageModel.STOP);
							currTrgLength++;
						}
						double lmScore = scoreLm(lm.getOrder(), currLmContextLength, lmContextBuf, currTrgLength, lm);
						score += lmScore;
						if (score > max) {
							best = translation;
							max = score;
							bestLmContextBuf = CollectionUtils.copyOf(lmContextBuf, lmContextBuf.length);
							bestLmContextLength = currLmContextLength + translation.english.indexedEnglish.length;
						}
					}
				}
			}
			ret.add(best);
			totalScore += max;
			currLmContextLength = bestLmContextLength;
			lmContextBuf = bestLmContextBuf;
			assert best != null;
			start += best.getForeignLength();
		}
		double explicitScore = Decoder.StaticMethods.scoreHypothesis(ret, lm, dm);
		if (Math.abs(explicitScore - totalScore) > 1e-4) {
			System.err.println("Warning: score calculated during decoding (" + totalScore + ") does not match explicit scoring (" + explicitScore + ")");
		}
		return ret;
	}

	private static double scoreLm(final int lmOrder, final int prevLmStateLength, final int[] lmStateBuf, final int totalTrgLength, final NgramLanguageModel lm) {
		double score = 0.0;

		if (prevLmStateLength < lmOrder - 1) {
			for (int i = 1; prevLmStateLength + i < lmOrder; ++i) {
				final double lmProb = lm.getNgramLogProbability(lmStateBuf, 0, prevLmStateLength + i);
				score += lmProb;
			}
		}
		for (int i = 0; i <= totalTrgLength - lmOrder; ++i) {
			final double lmProb = lm.getNgramLogProbability(lmStateBuf, i, i + lmOrder);
			score += lmProb;
		}
		return score;
	}
}
