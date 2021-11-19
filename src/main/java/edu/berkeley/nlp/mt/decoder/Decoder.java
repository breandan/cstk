package edu.berkeley.nlp.mt.decoder;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;
import edu.berkeley.nlp.util.BoundedList;
import edu.berkeley.nlp.util.StringIndexer;

public interface Decoder
{

	/**
	 * Decodes a sentence and returns a list of phrase pairs which
	 * representation a translation. These phrase pairs should be in
	 * English-side order.
	 * 
	 * @param frenchSentence
	 * @return
	 */
	public List<ScoredPhrasePairForSentence> decode(List<String> frenchSentence);

	public static class StaticMethods
	{

		/**
		 * Extracts the English side of a translation from a list of phrase
		 * pairs.
		 * 
		 * @param translation
		 * @return
		 */
		public static List<String> extractEnglish(List<ScoredPhrasePairForSentence> translation) {
			List<String> result = new ArrayList<String>();
			for (ScoredPhrasePairForSentence trans : translation) {
				result.addAll(trans.getEnglish());
			}
			return result;
		}

		/**
		 * Scores a translation by summing the translation costs with the
		 * language model cost for the English side of the translation.
		 * 
		 * @param hyp
		 * @param languageModel
		 * @param dm
		 * @return
		 */
		public static double scoreHypothesis(List<ScoredPhrasePairForSentence> hyp, NgramLanguageModel languageModel, DistortionModel dm) {
			double score = 0.0;
			double dmScore = 0.0;
			ScoredPhrasePairForSentence last = null;
			for (ScoredPhrasePairForSentence s : hyp) {
				score += s.score;
				final int lastEnd = last == null ? 0 : last.getEnd();
				final double distortionScore = dm.getDistortionScore(lastEnd, s.getStart());
				dmScore += distortionScore;
				last = s;
			}
			score += scoreSentenceWithLm(extractEnglish(hyp), languageModel, EnglishWordIndexer.getIndexer());
			score += dmScore;
			return score;
		}

		/**
		 * Scores an English sentence with a language model.
		 * 
		 * @param sentence
		 * @param lm
		 * @param lexIndexer
		 * @return
		 */
		public static double scoreSentenceWithLm(List<String> sentence, NgramLanguageModel lm, StringIndexer lexIndexer) {
			List<String> sentenceWithBounds = new BoundedList<String>(sentence, NgramLanguageModel.START, NgramLanguageModel.STOP);

			int lmOrder = lm.getOrder();
			double sentenceScore = 0.0;
			for (int i = 1; i < lmOrder - 1 && i <= sentenceWithBounds.size() + 1; ++i) {
				final List<String> ngram = sentenceWithBounds.subList(-1, i);
				int[] ngramArray = StaticMethods.toArray(ngram, lexIndexer);
				final double scoreNgram = lm.getNgramLogProbability(ngramArray, 0, ngramArray.length);
				sentenceScore += scoreNgram;
			}
			for (int i = lmOrder - 1; i < sentenceWithBounds.size() + 2; ++i) {
				final List<String> ngram = sentenceWithBounds.subList(i - lmOrder, i);
				int[] ngramArray = StaticMethods.toArray(ngram, lexIndexer);
				final double scoreNgram = lm.getNgramLogProbability(ngramArray, 0, ngramArray.length);
				sentenceScore += scoreNgram;
			}
			return sentenceScore;
		}

		/**
		 * @param ngram
		 * @param lexIndexer
		 * @return
		 */
		private static int[] toArray(final List<String> ngram, StringIndexer lexIndexer) {
			int[] ngramArray = new int[ngram.size()];
			for (int w = 0; w < ngramArray.length; ++w) {
				ngramArray[w] = lexIndexer.addAndGetIndex(ngram.get(w));
			}
			return ngramArray;
		}

	}

}
