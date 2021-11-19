package edu.berkeley.nlp.mt;

import edu.berkeley.nlp.util.Counter;

public interface NgramMultiset {
	public Counter<String> getNgrams(int zeroIndexed);
	public double getLength();
}
