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
public class IBMModel2Aligner extends IBMModel1Aligner implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;

  private static final int NUM_ITERATIONS = 15;
  private CounterMap<String,String> sourceTargetCounts = new CounterMap<String, String>();
  private Counter<String> targetCounts = new Counter<String>();
  private CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>> targetSourceIndexAndLenthCounts =
      new CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>>();
  private CounterMap<Integer, Integer> lengthCounts = new CounterMap<Integer, Integer>();
  private CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>> qCounters =
      new CounterMap<Pair<Integer, Integer>, Pair<Integer, Integer>>();

  private Set<String> sourceVocab = new HashSet<String>();
  private double initUniformDist;
  private static final String NULL_WORD = "nullWord";
  private static final int NULL_INDEX = -1;

  public Alignment align(SentencePair sentencePair) {
    Alignment alignment = new Alignment();
    List<String> sourceSentence = sentencePair.getSourceWords();
    List<String> targetSentence = sentencePair.getTargetWords();
    int m = sourceSentence.size();
    int l = targetSentence.size();

    for (int i = 0; i < m; i++) {
      double max = 0;
      String source = sourceSentence.get(i);
      int targetIndex = 0;
      for (int j = 0; j < l; j++) {
        String target = targetSentence.get(j);
        double prob = sourceTargetProbs.getCount(source, target) * getQ(j, i, l, m);
        if (prob > max) {
          max = prob;
          targetIndex = j;
        }
      }
      // Account for NULL word. Try taking this out.
      if ((sourceTargetProbs.getCount(source, NULL_WORD) * getQ(NULL_INDEX, i, l, m)) < max) {
        alignment.addPredictedAlignment(targetIndex, i);
      }
    }
    return alignment;
  }

  public void train(List<SentencePair> trainingPairs) {
    initStructures(trainingPairs);

    for (int k = 0; k < NUM_ITERATIONS; k++) {
      for (SentencePair pair : trainingPairs) {
        List<String> sourceSentence = pair.getSourceWords();
        int m = sourceSentence.size();
        List<String> targetSentence = pair.getTargetWords();
        int l = targetSentence.size();

        for (int i = 0; i < m; i++) {
          String source = sourceSentence.get(i);
          double denominator = getDenominator(source, targetSentence, i, m);
          for (int j = 0; j < l; j++) {
            String target = targetSentence.get(j);
            double prob = (sourceTargetProbs.getCount(source, target) * getQ(j, i, l, m)) / denominator;

            sourceTargetCounts.incrementCount(source, target, prob);
            targetCounts.incrementCount(target, prob);
            incrementTargetSourceIndexAndLenthCounts(j, i, l, m, prob);
            lengthCounts.incrementCount(l, m, prob);
          }

          // Update NULL word
          double prob = sourceTargetProbs.getCount(source, NULL_WORD) / denominator;
          sourceTargetCounts.incrementCount(source, NULL_WORD, prob);

          // This line might not be correct
          targetCounts.incrementCount(NULL_WORD, prob);
        }
      }
      renormalize(trainingPairs);
    }
  }

  private void renormalize(List<SentencePair> trainingPairs) {
    for (SentencePair pair : trainingPairs) {
      List<String> sourceSentence = pair.getSourceWords();
      int m = sourceSentence.size();
      List<String> targetSentence = pair.getTargetWords();
      int l = targetSentence.size();

      for (int i = 0; i < m; i++) {
        String source = sourceSentence.get(i);
        for (int j = 0; j < l; j++) {
          String target = targetSentence.get(j);
          double curCount = sourceTargetProbs.getCount(source, target);
          if (curCount != 0.0) {
            double newSourceTargetProb = sourceTargetCounts.getCount(source, target) / targetCounts.getCount(target);
            sourceTargetProbs.setCount(source, target, newSourceTargetProb);

            double newQValue = (getTargetSourceIndexAndLenthCounts(j, i, l, m)) / (lengthCounts.getCount(l, m));
            setQ(j, i, l, m, newQValue);
          }
        }
      }
    }
  }

  // Returns the denominator of the IBM Model 1 Formula.
  // A sum over all j for t(f(i)|e(j))
  private double getDenominator(String source, List<String> targetSentence, int i, int m) {
    double denominator = 0;
    int l = targetSentence.size();
    for (int j = 0; j < l; j++) {
      String target = targetSentence.get(j);
      denominator += sourceTargetProbs.getCount(source, target) * getQ(j, i, l, m);
    }
    return denominator + sourceTargetProbs.getCount(source, NULL_WORD) * getQ(NULL_INDEX, i, l, m);
  }

  private void initStructures(List<SentencePair> pairs) {
    super.train(pairs);

    for (SentencePair pair : pairs) {
      // Build vocab for source language.
      for (String source : pair.getSourceWords()) {
        sourceVocab.add(source);
      }
    }
    initUniformDist = 1.0 / (sourceVocab.size() + 1);

    for (SentencePair pair : pairs) {
      List<String> sourceSentence = pair.getSourceWords();
      int m = sourceSentence.size();
      List<String> targetSentence = pair.getTargetWords();
      int l = targetSentence.size();

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < l; j++) {
          qCounters.setCount(new Pair(j, i), new Pair(l, m), initUniformDist);
        }
        qCounters.setCount(new Pair(NULL_INDEX, i), new Pair(l,m), initUniformDist);
      }
    }
  }

  private double getQ(int j, int i, int l, int m) {
    return qCounters.getCount(new Pair(j, i), new Pair(l, m));
  }

  private void setQ(int j, int i, int l, int m, double value) {
    qCounters.setCount(new Pair(j, i), new Pair(l, m), value);
  }

  private void incrementQ(int j, int i, int l, int m, double increment) {
    qCounters.incrementCount(new Pair(j, i), new Pair(l, m), increment);
  }

  private double getTargetSourceIndexAndLenthCounts(int j, int i, int l, int m) {
    return targetSourceIndexAndLenthCounts.getCount(new Pair(j, i), new Pair(l, m));
  }

  private void incrementTargetSourceIndexAndLenthCounts(int j, int i, int l, int m, double count) {
    targetSourceIndexAndLenthCounts.incrementCount(new Pair(j, i), new Pair(l, m), count);
  }
}
