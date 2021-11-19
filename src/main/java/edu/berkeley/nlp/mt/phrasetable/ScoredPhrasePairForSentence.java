package edu.berkeley.nlp.mt.phrasetable;

import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.util.StrUtils;

/**
 * Stores a phrase pair together with a score.
 * 
 * @author adampauls
 * 
 */
public class ScoredPhrasePairForSentence
{

	public EnglishPhrase english;

	public double score;

	private List<String> foreignSentence;

	private int start;

	private int end;

	public ScoredPhrasePairForSentence(EnglishPhrase phrase, double score, List<String> foreignSentence, int start, int end) {
		this.english = phrase;
		this.score = score;
		this.foreignSentence = foreignSentence;
		this.start = start;
		this.end = end;
	}

	@Override
	public String toString() {
		return "[ScoredPhrasePair fr=" + StrUtils.join(getForeign()) + ", en=" + StrUtils.join(getEnglish()) + ", score=" + score + "]";

	}

	public List<String> getEnglish() {
		return Arrays.asList(english.getTarget());
	}

	public List<String> getForeign() {
		return foreignSentence.subList(start, end);
	}

	public int getForeignLength() {
		return end - start;
	}

	public int getStart() {
		return start;
	}

	public int getEnd() {
		return end;
	}

}