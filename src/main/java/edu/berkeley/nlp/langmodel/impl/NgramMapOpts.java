package edu.berkeley.nlp.langmodel.impl;

import java.lang.reflect.Field;

public class NgramMapOpts
{

	/**
	 * Number of longs (8 bytes) used as a "block" for variable length
	 * compression.
	 */
	public int compressedBlockSize = 16;

	/**
	 * Use variable-length compression
	 */
	public boolean compress = false;

	/**
	 * 
	 */
	public int suffixRadix = 6;

	public boolean buildIndex = true;

	public boolean directBinarySearch = false;

	public boolean useHash = false;

	public boolean timersOn = false;

	public boolean interpolationSearch = false;

	public boolean averageInterpolate = false;

	public boolean cacheSuffixes = false;

	public boolean storeWordsImplicitly = false;

	public double maxLoadFactor = 0.7;

	public boolean countDeltas = false;

	public int numGoogleLoadThreads = 0;

	public boolean skipCompressingVals;

	public boolean absDeltas = false;

	public int valueRadix = 6;

	public boolean useHuffman = false;

	public int miniIndexNum = -1;

	public boolean backEndCache = false;

	public int huffmanCountCutoff = 1 << 16;

	public boolean skipLinearSearch = false;

	public boolean storePrefixIndexes = true;

	public boolean logJoshuaLmRequests = false;

	public boolean quadraticProbing = false;

	public float unknownWordLogProb = -100.0f;

	public NgramMapOpts copy() {
		NgramMapOpts c = new NgramMapOpts();
		for (Field f : getClass().getFields()) {
			try {
				f.set(c, f.get(this));
			} catch (IllegalArgumentException e) {
				// TODO Auto-generated catch block
				throw new RuntimeException(e);

			} catch (IllegalAccessException e) {
				// TODO Auto-generated catch block
				throw new RuntimeException(e);

			}
		}
		return this;

	}
}