package edu.berkeley.nlp.langmodel;

import edu.berkeley.nlp.util.StringIndexer;

/**
 * Class responsible for maintaining a global mapping between English words and
 * unique integers.
 * 
 * @author adampauls
 * 
 */
public class EnglishWordIndexer
{

	private static StringIndexer indexer = new StringIndexer();

	public static StringIndexer getIndexer() {
		return indexer;
	}

}
