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
public class ScoredPhrase
{

	public EnglishPhrase english;

	public double score;

	public ScoredPhrase(EnglishPhrase phrase, double score) {
		this.english = phrase;
		this.score = score;

	}

	@Override
	public String toString() {
		return "[ScoredPhrase en=" + StrUtils.join(getEnglish()) + ", score=" + score + "]";

	}

	public List<String> getEnglish() {
		return Arrays.asList(english.getTarget());
	}

}