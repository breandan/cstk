package edu.berkeley.nlp.mt.phrasetable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * This class pre-computes the phrase pairs which apply to a given (foreign)
 * sentence. To access these phrase pairs, call
 * getScoreSortedTranslationsForSpan. This will return a list sorted in
 * descending order of score (i.e. best first). This list will have at most
 * PhraseTable.getMaxNumTranslations().
 * 
 * @author adampauls
 * 
 */
public class PhraseTableForSentence
{

	private List<String> sentence;

	private List<ScoredPhrasePairForSentence>[][] rulesSortedByScore;

	private PhraseTable phraseTable;

	public PhraseTableForSentence(PhraseTable phraseTable, List<String> foreignSentence) {
		this.sentence = foreignSentence;
		this.phraseTable = phraseTable;
		init();
	}

	private void init() {
		rulesSortedByScore = new List[sentence.size()][phraseTable.getMaxPhraseSize()];
		int currIndex = 0;
		for (int i = 0; i < sentence.size(); ++i) {
			for (int j = i + 1; j <= i + phraseTable.getMaxPhraseSize(); ++j) {
				if (j > sentence.size()) break;
				final List<String> subList = sentence.subList(i, j);
				final String[] srcArray = subList.toArray(new String[subList.size()]);
				final List<ScoredPhrase> c = phraseTable.getTranslationsFor(subList);
				List<ScoredPhrase> scoredTranslations = c == null ? getFakeTranslations(sentence, i, j) : Collections.unmodifiableList(c);
				if (scoredTranslations == null) continue;
				List<ScoredPhrasePairForSentence> scoredIndexedTranslations = new ArrayList<ScoredPhrasePairForSentence>(scoredTranslations.size());
				for (ScoredPhrase t : scoredTranslations) {
					if (scoredIndexedTranslations.size() < phraseTable.getMaxNumTranslations()) {
						final ScoredPhrasePairForSentence e = new ScoredPhrasePairForSentence(t.english, t.score, sentence, i, j);
						scoredIndexedTranslations.add(e);
					}
				}
				rulesSortedByScore[i][j - i - 1] = scoredIndexedTranslations;
			}
		}

	}

	/**
	 * Create a "fake" translations for foreign words we've never seen before.
	 * 
	 * @param sentence
	 * @param i
	 * @param j
	 * @return
	 */
	private static List<ScoredPhrase> getFakeTranslations(List<String> sentence, int i, int j) {
		if (j - i > 1) return null;
		ScoredPhrase fake = new ScoredPhrase(new EnglishPhrase(new String[] { sentence.get(i) }), 0.0);

		return Collections.singletonList(fake);
	}

	/**
	 * Returns a list of phrase pairs which apply for a given span of the
	 * foreign sentence. If the foreign sentence is [le, chat, dors, .], then
	 * getScoreSortedTranslationsForSpan(0,2) will return all translations of
	 * "le chat".
	 * 
	 * @param i
	 *            Begin of span (inclusive)
	 * @param j
	 *            End of span (exclusive)
	 * @return
	 */
	public List<ScoredPhrasePairForSentence> getScoreSortedTranslationsForSpan(int i, int j) {
		return rulesSortedByScore[i][j - i - 1];
	}

	public int getMaxPhraseLength() {
		return phraseTable.getMaxPhraseSize();
	}
}