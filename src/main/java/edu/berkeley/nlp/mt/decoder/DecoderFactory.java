package edu.berkeley.nlp.mt.decoder;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;

public interface DecoderFactory
{
	public Decoder newDecoder(PhraseTable tm, NgramLanguageModel lm, DistortionModel dm);

}
