package cs224n.wordaligner;

import cs224n.util.*;
import java.util.List;
import java.util.Set;
import java.util.HashSet;

/**
 * Simple word alignment baseline model that maps source positions to target
 * positions along the diagonal of the alignment grid.
 *
 * IMPORTANT: Make sure that you read the comments in the
 * cs224n.wordaligner.WordAligner interface.
 *
 * @author Dan Klein
 * @author Spence Green
 */
public class IBMModel1Aligner implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;

  private static final int NUM_ITERATIONS = 15;
  private CounterMap<String,String> sourceTargetCounts = new CounterMap<String, String>();
  public CounterMap<String, String> sourceTargetProbs = new CounterMap<String, String>();
  private Counter<String> targetCounts = new Counter<String>();
  private Set<String> sourceVocab = new HashSet<String>();
  private double initUniformDist;
  private static final String NULL_WORD = "nullWord";

  public Alignment align(SentencePair sentencePair) {
    Alignment alignment = new Alignment();
    List<String> sourceSentence = sentencePair.getSourceWords();
    List<String> targetSentence = sentencePair.getTargetWords();

    for (int i = 0; i < sourceSentence.size(); i++) {
      double max = 0;
      String source = sourceSentence.get(i);
      int targetIndex = 0;
      for (int j = 0; j < targetSentence.size(); j++) {
        String target = targetSentence.get(j);
        double prob = sourceTargetProbs.getCount(source, target);
        if (prob > max) {
          max = prob;
          targetIndex = j;
        }
      }
      // Account for NULL word.
      if ((sourceTargetProbs.getCount(source, NULL_WORD)) < max) {
        alignment.addPredictedAlignment(targetIndex, i);
      }
    }
    return alignment;
  }

  public void train(List<SentencePair> trainingPairs) {
    initStructures(trainingPairs);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
      for (SentencePair pair : trainingPairs) {
        List<String> sourceSentence = pair.getSourceWords();
        List<String> targetSentence = pair.getTargetWords();

        for (String source : sourceSentence) {
          double denominator = getDenominator(source, targetSentence);

          for (String target : targetSentence) {
            double prob = sourceTargetProbs.getCount(source, target) / denominator;
            sourceTargetCounts.incrementCount(source, target, prob);
            targetCounts.incrementCount(target, prob);
          }

          // Update NULL word
          double prob = sourceTargetProbs.getCount(source, NULL_WORD) / denominator;
          sourceTargetCounts.incrementCount(source, NULL_WORD, prob);

          // This line might not be correct
          targetCounts.incrementCount(NULL_WORD, prob);
        }
      }
      renormalize();
    }
  }

  private void renormalize() {
    for (String source : sourceTargetProbs.keySet()) {
      for (String target : targetCounts.keySet()) {
        double curCount = sourceTargetProbs.getCount(source, target);
        if (curCount != 0.0) {
          double newSourceTargetProb = sourceTargetCounts.getCount(source, target) / targetCounts.getCount(target);
          sourceTargetProbs.setCount(source, target, newSourceTargetProb);
        }
      }
    }
  }

  // Returns the denominator of the IBM Model 1 Formula.
  // A sum over all j for t(f(i)|e(j))
  private double getDenominator(String source, List<String> targetSentence) {
    double denominator = 0;
    for (String target : targetSentence) {
      denominator += sourceTargetProbs.getCount(source, target);
    }
    return denominator + sourceTargetProbs.getCount(source, NULL_WORD);
  }

  private void initStructures(List<SentencePair> pairs) {
    for (SentencePair pair : pairs) {
      // Build vocab for source language.
      for (String source : pair.getSourceWords()) {
        sourceVocab.add(source);
      }
    }

    initUniformDist = 1.0 / (sourceVocab.size() + 1);

    for (SentencePair pair : pairs) {
      for (String source : pair.getSourceWords()) {
        for (String target : pair.getTargetWords()) {
          // Uniformly init t(f|e) s.t. each entry = 1 / (size of source (french) vocab + 1)
          sourceTargetProbs.setCount(source, target, initUniformDist);
        }
        sourceTargetProbs.setCount(source, NULL_WORD, initUniformDist);
      }
    }
  }
}
