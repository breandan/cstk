package edu.berkeley.nlp.langmodel.impl;

import java.util.List;

import edu.berkeley.nlp.util.StringIndexer;

/**
 * Interface for language models which expose the internal context-encoding for
 * more efficient queries. (Note: language model implementations may internally
 * use a context-encoding without implementing this interface).
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public interface ContextEncodedNgramLanguageModel extends EfficientNGramLanguageModel
{

	/**
	 * Simple class for returning context offsets
	 * 
	 * @author adampauls
	 * 
	 */
	public static class LmContextInfo
	{

		/**
		 * Offset of context (prefix) of an n-gram
		 */
		public long context = -1L;

		/**
		 * The (0-based) length of <code>context</code> (i.e.
		 * <code>order == 0</code> iff <code>context</code> refers to a
		 * unigram).
		 */
		public int order = -1;

	}

	/**
	 * 
	 * @param context
	 *            Offset of context (prefix) of an n-gram
	 * @param contextOrder
	 *            The (0-based) length of <code>context</code> (i.e.
	 *            <code>order == 0</code> iff <code>context</code> refers to a
	 *            unigram).
	 * @param word
	 *            Last word of the n-gram
	 * @param outputContext
	 *            This is an output parameter. It will be to the offset of the
	 *            longest suffix of the input n-gram which is contained in the
	 *            model.
	 * 
	 * @return
	 */
	public float getLogProb(long context, int contextOrder, int word, LmContextInfo outputContext);

	/**
	 * Same as getLogProb(long context, int contextOrder, int word,
	 * LmContextInfo outputContext), except that the input n-gram is represent
	 * as an explicit int[] array
	 * 
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @param outputContext
	 * @return
	 */
	public float getLogProb(int[] ngram, int startPos, int endPos, LmContextInfo outputContext);

	public float getLogProb(List<String> ngram, LmContextInfo outputContext);

	public static class DefaultImplementations
	{

		public static float getLogProb(List<String> ngram, LmContextInfo contextOutput, ContextEncodedNgramLanguageModel lm) {
			int[] ints = new int[ngram.size()];
			final StringIndexer wordIndexer = lm.getWordIndexer();
			for (int i = 0; i < ngram.size(); ++i) {
				ints[i] = wordIndexer.addAndGetIndex(ngram.get(i));
			}
			return lm.getLogProb(ints, 0, ints.length, contextOutput);

		}

		public static float getLogProb(List<String> ngram, ContextEncodedNgramLanguageModel lm) {
			return getLogProb(ngram, null, lm);
		}

		public static float getLogProb(int[] ngram, int startPos, int endPos, ContextEncodedNgramLanguageModel lm) {
			return lm.getLogProb(ngram, startPos, endPos, null);
		}

	}

}
