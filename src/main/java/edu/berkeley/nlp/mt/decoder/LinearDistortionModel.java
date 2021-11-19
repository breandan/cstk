package edu.berkeley.nlp.mt.decoder;

public class LinearDistortionModel implements DistortionModel
{
	private double weight;

	private int distortionLimit;

	public LinearDistortionModel(int distortionLimit, double weight) {
		this.distortionLimit = distortionLimit;
		this.weight = weight;
	}

	/* (non-Javadoc)
	 * @see edu.berkeley.nlp.assignments.assign2.solutions.DistortionModel#getDistortionLimit()
	 */
	public int getDistortionLimit() {
		return this.distortionLimit;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.assignments.assign2.solutions.DistortionModel#
	 * getDistortionScore(int, int)
	 */
	public double getDistortionScore(int endPrev, int beginCurr) {
		int dist = Math.abs(beginCurr - endPrev);
		if (dist > distortionLimit) { return Double.NEGATIVE_INFINITY; }
		return weight * dist;
	}
}
