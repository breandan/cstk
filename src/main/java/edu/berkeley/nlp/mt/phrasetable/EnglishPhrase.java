package edu.berkeley.nlp.mt.phrasetable;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;

/**
 * A simple container class for the English side of a phrase pair.
 * 
 * @author adampauls
 * 
 */
public class EnglishPhrase
{

	public EnglishPhrase(String[] english) {
		this.indexedEnglish = new int[english.length];
		for (int i = 0; i < indexedEnglish.length; ++i)
			indexedEnglish[i] = EnglishWordIndexer.getIndexer().addAndGetIndex(english[i]);
	}

	public EnglishPhrase(int[] indexedEnglish) {
		this.indexedEnglish = indexedEnglish;
	}

	public int[] indexedEnglish;

	public String[] getTarget() {
		String[] trg = new String[indexedEnglish.length];
		for (int i = 0; i < indexedEnglish.length; ++i)
			trg[i] = EnglishWordIndexer.getIndexer().get(indexedEnglish[i]);
		return trg;
	}

}