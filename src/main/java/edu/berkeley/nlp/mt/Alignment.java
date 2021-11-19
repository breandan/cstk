package edu.berkeley.nlp.mt;

import java.util.HashSet;
import java.util.Set;

import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.util.Pair;

/**
   * Alignments serve two purposes, both to indicate your system's guessed
   * alignment, and to hold the gold standard alignments.  Alignments map index
   * pairs to one of three values, unaligned, possibly aligned, and surely
   * aligned.  Your alignment guesses should only contain sure and unaligned
   * pairs, but the gold alignments contain possible pairs as well.
   *
   * To build an alignemnt, start with an empty one and use
   * addAlignment(i,j,true).  To display one, use the render method.
   */
  public class Alignment {
    Set<Pair<Integer, Integer>> sureAlignments;
    Set<Pair<Integer, Integer>> possibleAlignments;

    public boolean containsSureAlignment(int englishPosition, int frenchPosition) {
      return sureAlignments.contains(new Pair<Integer, Integer>(englishPosition, frenchPosition));
    }

    public boolean containsPossibleAlignment(int englishPosition, int frenchPosition) {
      return possibleAlignments.contains(new Pair<Integer, Integer>(englishPosition, frenchPosition));
    }

    public void addAlignment(int englishPosition, int frenchPosition, boolean sure) {
      Pair<Integer, Integer> alignment = new Pair<Integer, Integer>(englishPosition, frenchPosition);
      if (sure)
        sureAlignments.add(alignment);
      possibleAlignments.add(alignment);
    }

    public Alignment() {
      sureAlignments = new HashSet<Pair<Integer, Integer>>();
      possibleAlignments = new HashSet<Pair<Integer, Integer>>();
    }

    public static String render(Alignment alignment, SentencePair sentencePair) {
      return render(alignment, alignment, sentencePair);
    }

    public static String render(Alignment reference, Alignment proposed, SentencePair sentencePair) {
      StringBuilder sb = new StringBuilder();
      for (int frenchPosition = 0; frenchPosition < sentencePair.getFrenchWords().size(); frenchPosition++) {
        for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
          boolean sure = reference.containsSureAlignment(englishPosition, frenchPosition);
          boolean possible = reference.containsPossibleAlignment(englishPosition, frenchPosition);
          char proposedChar = ' ';
          if (proposed.containsSureAlignment(englishPosition, frenchPosition))
            proposedChar = '#';
          if (sure) {
            sb.append('[');
            sb.append(proposedChar);
            sb.append(']');
          } else {
            if (possible) {
              sb.append('(');
              sb.append(proposedChar);
              sb.append(')');
            } else {
              sb.append(' ');
              sb.append(proposedChar);
              sb.append(' ');
            }
          }
        }
        sb.append("| ");
        sb.append(sentencePair.getFrenchWords().get(frenchPosition));
        sb.append('\n');
      }
      for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
        sb.append("---");
      }
      sb.append("'\n");
      boolean printed = true;
      int index = 0;
      while (printed) {
        printed = false;
        StringBuilder lineSB = new StringBuilder();
        for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
          String englishWord = sentencePair.getEnglishWords().get(englishPosition);
          if (englishWord.length() > index) {
            printed = true;
            lineSB.append(' ');
            lineSB.append(englishWord.charAt(index));
            lineSB.append(' ');
          } else {
            lineSB.append("   ");
          }
        }
        index += 1;
        if (printed) {
          sb.append(lineSB);
          sb.append('\n');
        }
      }
      return sb.toString();
    }
  }