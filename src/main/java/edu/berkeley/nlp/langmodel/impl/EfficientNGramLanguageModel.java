package edu.berkeley.nlp.langmodel.impl;

import java.util.List;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.util.BoundedList;
import edu.berkeley.nlp.util.StringIndexer;

/**
 * 
 * @author adampauls Top-level interface for an n-gram language model.
 * 
 * @param <W>
 *            A type representing words in the language. Can be a
 *            <code>String</code>, or something more complex if needed
 */
public interface EfficientNGramLanguageModel
{

	/**
	 * Maximum size of n-grams stored by the model.
	 * 
	 * @return
	 */
	public int getLmOrder();

	/**
	 * Each LM must have a WordIndexer which assigns integer IDs to each word W
	 * in the language.
	 * 
	 * @return
	 */
	public StringIndexer getWordIndexer();

	// XXX: ned to make @see work here
	/**
	 * Convenience method -- the list is first converted to an int[]
	 * representation. This is general inefficient, and user code should
	 * directly provide int[] arrays, or even better, provide (context, word)
	 * pairs.
	 * 
	 * @see getLogProb(NgramPrefix inputPrefix, int word, NgramPrefix
	 *      outputPrefixIndex)
	 * @param ngram
	 * @return
	 */
	public float getLogProb(List<String> ngram);

	/**
	 * Calculate language model score of an n-gram.
	 * 
	 * @param ngram
	 *            array of words in integer representation
	 * @param startPos
	 *            start of the portion of the array to be read
	 * @param endPos
	 *            end of the portion of the array to be read.
	 * @return
	 */
	public float getLogProb(int[] ngram, int startPos, int endPos);

	/**
	 * Equivalent to getLogProb(ngram, 0, ngram.length).
	 */
	public float getLogProb(int[] ngram);

	/**
	 * Scores sequence possibly containing multiple n-grams, but not a complete
	 * sentence.
	 * 
	 * @return
	 */
	public float scoreSequence(List<String> sequence);

	/**
	 * Scores a complete sentence, taking appropriate care with the start- and
	 * end-of-sentence symbols.
	 * 
	 * @return
	 */
	public float scoreSentence(List<String> sentence);

	public static class DefaultImplementations
	{

		public static float scoreSentence(List<String> sentence, EfficientNGramLanguageModel lm) {
			List<String> sentenceWithBounds = new BoundedList<String>(sentence, NgramLanguageModel.START, NgramLanguageModel.STOP);

			int lmOrder = lm.getLmOrder();
			float sentenceScore = 0.0f;
			for (int i = 1; i < lmOrder - 1 && i <= sentenceWithBounds.size() + 1; ++i) {
				final List<String> ngram = sentenceWithBounds.subList(-1, i);
				final float scoreNgram = lm.getLogProb(ngram);
				sentenceScore += scoreNgram;
			}
			for (int i = lmOrder - 1; i < sentenceWithBounds.size() + 2; ++i) {
				final List<String> ngram = sentenceWithBounds.subList(i - lmOrder, i);
				final float scoreNgram = lm.getLogProb(ngram);
				sentenceScore += scoreNgram;
			}
			return sentenceScore;
		}

		public static float getLogProb(int[] ngram, EfficientNGramLanguageModel lm) {
			return lm.getLogProb(ngram, 0, ngram.length);
		}

		public static float scoreSequence(List<String> sequence, EfficientNGramLanguageModel lm) {
			float sentenceScore = 0.0f;

			int lmOrder = lm.getLmOrder();
			for (int i = 0; i + lmOrder - 1 < sequence.size(); ++i) {
				final List<String> ngram = sequence.subList(i, i + lmOrder);
				final float scoreNgram = lm.getLogProb(ngram);
				sentenceScore += scoreNgram;
			}
			return sentenceScore;
		}

		public static float getLogProb(List<String> ngram, EfficientNGramLanguageModel lm) {
			int[] ints = new int[ngram.size()];
			final StringIndexer wordIndexer = lm.getWordIndexer();
			for (int i = 0; i < ngram.size(); ++i) {
				ints[i] = wordIndexer.addAndGetIndex(ngram.get(i));
			}
			return lm.getLogProb(ints, 0, ints.length);

		}

	}

}
