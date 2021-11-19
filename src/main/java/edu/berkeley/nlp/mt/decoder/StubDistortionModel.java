package edu.berkeley.nlp.mt.decoder;

public class StubDistortionModel implements DistortionModel
{

	public StubDistortionModel() {
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.assignments.assign2.solutions.DistortionModel#
	 * getDistortionLimit()
	 */
	public int getDistortionLimit() {
		return 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.assignments.assign2.solutions.DistortionModel#
	 * getDistortionScore(int, int)
	 */
	public double getDistortionScore(int endPrev, int beginCurr) {
		return 0.0;
	}
}
