package edu.mcgill.gymfs.inference

import ai.djl.Application
import ai.djl.modality.nlp.qa.QAInput
import ai.djl.repository.zoo.*
import ai.djl.training.util.ProgressBar
import org.slf4j.LoggerFactory

fun main() {
  val answer = BertQaInference.predict()
  BertQaInference.logger.info("Answer: {}", answer)
}

object BertQaInference {
  val logger = LoggerFactory.getLogger(BertQaInference::class.java)

  fun predict(): String {
    val question = "when did the cat arrive?"
    val paragraph = "when we were eating lunch, the cat arrived"
    val input = QAInput(question, paragraph)
    logger.info("Paragraph: {}", input.paragraph)
    logger.info("Question: {}", input.question)
    val criteria = Criteria.builder()
      .optApplication(Application.NLP.TEXT_EMBEDDING)
      .setTypes(QAInput::class.java, String::class.java)
      .optFilter("backbone", "bert")
      .optProgress(ProgressBar())
      .build()
    ModelZoo.loadModel(criteria).use { model ->
      model.newPredictor().use { predictor ->
        return predictor.predict(input)
      }
    }
  }
}