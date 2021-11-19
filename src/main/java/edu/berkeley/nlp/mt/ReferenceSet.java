package edu.berkeley.nlp.mt;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.berkeley.nlp.io.IOUtils;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.StrUtils;

/**
 * A collection of references for one sentence
 * 
 * @author denero
 */
public class ReferenceSet implements NgramMultiset
{

	private List<String> references;

	List<Counter<String>> ngrams;

	private int shortest;

	public ReferenceSet(List<String> references) {
		this(references, 4);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		int i = 0;
		for (String r : references) {
			sb.append("Reference " + i++ + ": " + r + "\n");
		}

		return sb.toString().trim();

	}

	public ReferenceSet(List<String> references, int maxNgramSize) {
		this.references = references;
		shortest = Integer.MAX_VALUE;
		ngrams = new ArrayList<Counter<String>>(maxNgramSize);
		for (String r : references) {
			List<String> words = tokenizeAndIntern(r);
			for (int len = 0; len < maxNgramSize; len++) {
				if (ngrams.size() <= len) ngrams.add(new Counter<String>());
				Counter<String> thisCtr = ngrams.get(len);
				Counter<String> rCtr = BleuScore.countNgrams(len, words);
				for (String s : rCtr.keySet()) {
					thisCtr.setCount(s, Math.max(thisCtr.getCount(s), rCtr.getCount(s)));
				}
			}
			shortest = Math.min(shortest, words.size());
		}
	}

	private static String[] tokenize(String s) {
		String[] words = s.trim().split("\\s+");
		return words;
	}

	private static List<String> tokenizeAndIntern(String sentence) {
		String[] raw = tokenize(sentence);
		List<String> words = new ArrayList<String>(raw.length);
		for (int i = 0; i < raw.length; i++) {
			words.add(raw[i].intern());
		}
		return words;
	}

	public ReferenceSet(String s) {
		this(Collections.singletonList(s));
	}

	public Counter<String> getNgrams(int zeroIndexed) {
		return ngrams.get(zeroIndexed);
	}

	public double getLength() {
		return shortest;
	}

	public List<String> getReferences() {
		return references;
	}

	public static ReferenceSet referenceSetFromSentencePair(SentencePair sentencePair, String englishExtension) {
		List<String> references = new ArrayList<String>();

		String reference1 = StrUtils.join(sentencePair.getEnglishWords());
		references.add(reference1);

		return new ReferenceSet(references);

	}

	/**
	 * Reads the next reference from a set of input files
	 * 
	 * @return
	 * @throws IOException
	 */
	private static ReferenceSet nextReference(List<BufferedReader> inputFiles) throws IOException {
		List<String> sents = new ArrayList<String>(inputFiles.size());
		for (BufferedReader reader : inputFiles) {
			sents.add(reader.readLine());
		}
		return new ReferenceSet(sents);
	}

	private static List<ReferenceSet> readReferences(List<BufferedReader> inputFiles, int numSent) throws IOException {
		ArrayList<ReferenceSet> refs = new ArrayList<ReferenceSet>();
		while (inputFiles.get(0).ready() && refs.size() < numSent) {
			refs.add(nextReference(inputFiles));
		}
		return refs;
	}

	public static List<ReferenceSet> readReferences(String prefix, String suffix, int maxRefs, int numSent) {
		ArrayList<BufferedReader> inputFiles = new ArrayList<BufferedReader>();
		try {
			for (int ref = 0; ref < maxRefs; ref++) {
				String name = prefix + ref + suffix;
				//				Logger.logss("Attempting to open reference: " + name);
				BufferedReader r = IOUtils.openIn(name);
				if (r == null) break;
				inputFiles.add(r);
			}
			if (inputFiles.size() == 0) return null;

			return readReferences(inputFiles, numSent);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

}
