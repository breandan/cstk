package edu.berkeley.nlp.mt.phrasetable;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import edu.berkeley.nlp.io.IOUtils;
import edu.berkeley.nlp.mt.decoder.Logger;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Counter;

/**
 * Stores phrase pairs and their scores. Inside a decoder, you should call
 * initialize() to get back an object which returns translations for a specific
 * sentence.
 * 
 * @author adampauls
 * 
 */
public class PhraseTable
{

	private int maxPhraseSize;

	private int maxNumTranslations;

	private static class PhrasePair
	{
		/**
		 * @param foreign
		 * @param english
		 */
		public PhrasePair(String[] foreign, EnglishPhrase english) {
			super();
			this.foreign = foreign;
			this.english = english;
		}

		String[] foreign;

		EnglishPhrase english;
	}

	Map<List<String>, List<ScoredPhrase>> table;

	public static final String[] MOSES_FEATURE_NAMES = new String[] { "P(f|e)", "lex(f|e)", "P(e|f)", "lex(e|f)", "bias", "wordBonus" };

	/**
	 * 
	 * @param maxPhraseSize
	 *            The maximum length of either side of a phrase
	 * @param maxNumTranslations
	 *            The maximum number of translations per foreign span.
	 */
	public PhraseTable(int maxPhraseSize, int maxNumTranslations) {
		this.maxPhraseSize = maxPhraseSize;
		this.maxNumTranslations = maxNumTranslations;
	}

	public PhraseTableForSentence initialize(List<String> sentence) {
		return new PhraseTableForSentence(this, sentence);
	}

	public int getMaxPhraseSize() {
		return maxPhraseSize;
	}

	public int getMaxNumTranslations() {
		return maxNumTranslations;
	}

	public void readFromFile(String file, Counter<String> featureWeights) {

		initStorage();
		Logger.startTrack("Reading phrase table from file " + file);
		int l = 0;

		try {
			for (String line : CollectionUtils.iterable(IOUtils.lineIterator(file))) {
				l++;
				if (l % 100000 == 0) System.out.println("Line " + l);
				float[] features = new float[6];
				PhrasePair phrasePair = readMosesRule(line, features);

				if (phrasePair.english.indexedEnglish.length > maxPhraseSize) continue;
				if (phrasePair.foreign.length > maxPhraseSize) continue;
				ScoredPhrase t = new ScoredPhrase(phrasePair.english, getFeatureCounter(features).dotProduct(featureWeights));

				addTranslation(t, Arrays.asList(phrasePair.foreign));

			}
		} catch (IOException e) {
			throw new RuntimeException(e);

		}

		sortTranslations();
		Logger.endTrack();
	}

	List<ScoredPhrase> getTranslationsFor(List<String> subList) {
		return table.get(subList);
	}

	private Counter<String> getFeatureCounter(float[] features) {
		Counter<String> ret = new Counter<String>();
		for (int i = 0; i < features.length; ++i) {
			ret.setCount(MOSES_FEATURE_NAMES[i], features[i]);
		}
		return ret;
	}

	private PhrasePair readMosesRule(String ruleString, float[] features) {
		String[] parts = ruleString.trim().split("\\|\\|\\|");
		assert (parts.length == 3 || parts.length == 5);
		if (parts.length == 5) parts[2] = parts[4];
		final String[] srcArray = parts[0].trim().split(" ");
		final String[] trgArray = parts[1].trim().split(" ");
		intern(srcArray);
		intern(trgArray);

		String[] featStrings = parts[2].trim().split("\\s+");
		for (int i = 0; i < featStrings.length; i++) {

			try {
				Double val = Double.parseDouble(featStrings[i]);
				if (val.isInfinite() || val.isNaN()) {
					Logger.warn("Non-finite feature: " + featStrings[i]);
					continue;
				}
				val = -Math.log(val);

				features[i] = val.floatValue();
			} catch (NumberFormatException n) {
				Logger.warn("Feature syntax error: " + featStrings[i]);
			}
		}
		features[5] = trgArray.length;
		return new PhrasePair(srcArray, new EnglishPhrase(trgArray));

	}

	private void intern(String[] a) {
		for (int i = 0; i < a.length; ++i)
			a[i] = a[i].intern();
	}

	private void initStorage() {
		table = new HashMap<List<String>, List<ScoredPhrase>>();
	}

	private void addTranslation(ScoredPhrase t, List<String> foreign) {
		CollectionUtils.addToValueList(table, foreign, t);
	}

	private void sortTranslations() {

		for (Entry<List<String>, List<ScoredPhrase>> entry : table.entrySet()) {

			Collections.sort(entry.getValue(), new Comparator<ScoredPhrase>()
			{

				public int compare(ScoredPhrase o1, ScoredPhrase o2) {
					return Double.compare(o2.score, o1.score);
				}
			});
		}
	}

}