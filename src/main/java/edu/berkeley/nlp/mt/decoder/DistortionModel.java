package edu.berkeley.nlp.mt.decoder;

public interface DistortionModel
{

	/**
	 * Returns the distortion limit. If the distortion limit is 0, then no
	 * distortion is permitted. If the distortion limit is n, then the ith
	 * foreign word can be translated only if all of the first (i-n) foreign
	 * words have already been consumed.
	 * 
	 * @return
	 */
	public abstract int getDistortionLimit();

	/**
	 * Interface for a distortion score.
	 * 
	 * @param endPrev
	 *            The foreign-side index after the end of the previous phrase
	 *            translated. Should be 0 at the beginning. For linear
	 *            distortion, if this is the same as beginCurr, you incur no
	 *            distortion, otherwise you incur linear distortion with the
	 *            given weight.
	 * @param beginCurr
	 *            The foreign-side index of the beginning of the phrase about to
	 *            be translated. For example, if you translation the 2nd word of
	 *            a sentence after translating the 3rd, then beginCurr = 1
	 *            (because sentences are 0-indexed) and endPrev = 3. If you
	 *            translate the 3rd after the 2nd, then beginCurr = 2 and
	 *            endPrev = 2.
	 * @return The distortion score
	 */
	public abstract double getDistortionScore(int endPrev, int beginCurr);

}