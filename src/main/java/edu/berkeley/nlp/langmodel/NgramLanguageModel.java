package edu.berkeley.nlp.langmodel;

/**
 * Language models assign probabilities to n-grams.
 * 
 * @author Adam Pauls
 */
public interface NgramLanguageModel
{
	/**
	 * This symbol is a unique start word which should be prepended to all
	 * sentences when estimating probabilities.
	 */
	public static final String START = "<s>";

	/**
	 * This symbol is a unique start word which should be apppended to all
	 * sentences when estimating probabilities.
	 */
	public static final String STOP = "</s>";

	/**
	 * Maximum order of n-gram that will be scored by the model
	 * 
	 * @return
	 */
	int getOrder();

	/**
	 * Score the sequence of words in the ngram array over the subrange of the
	 * array specified by from and to. For example,
	 * getNgramLogProbability([17,15,18],1,3) should return the (log of)
	 * P(w_i=18 | w_{i-1} = 15). Anything outside the bounds from and to is
	 * ignored. (This choice of interface allows for efficient reuse of arrays
	 * inside the decoder).
	 * 
	 * The integers represent words (Strings) via the mapping giving by
	 * EnglishWordIndexer.getIndexer().
	 * 
	 * Note that even a trigram language model must score bigrams (e.g. at the
	 * beginning of a sentence), and so you should not assume that to == from +
	 * getOrder().
	 * 
	 * @param ngram
	 * @param from
	 * @param to
	 * @return
	 */
	double getNgramLogProbability(int[] ngram, int from, int to);

	/**
	 * Returns the count of an n-gram. We will call this function when testing
	 * your code.
	 * 
	 * @param ngram
	 * @return
	 */
	long getCount(int[] ngram);
}
