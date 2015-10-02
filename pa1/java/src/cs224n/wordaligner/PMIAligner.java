package cs224n.wordaligner;

import cs224n.util.*;
import java.util.List;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
public class PMIAligner implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;

  // from the training data.
  private Map<String, Integer> sourceWordCounts = new HashMap<String, Integer>();
  private Map<String, Integer> targetWordCounts = new HashMap<String, Integer>();

  // How many times did word f(i) and word e(j) occur together
  private CounterMap<String,String> sourceTargetCounts = new CounterMap<String, String>();

  private int totalNumPairs = 0;

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

        double score = getScore(source, target);
        if (score > max) {
          max = score;
          targetIndex = j;
        }
      }
      
      // Account for NULL word.
      if ((1.0 / totalNumPairs) < max) {
        alignment.addPredictedAlignment(targetIndex, i);
      }
    }
    return alignment;
  }

  private double getScore(String source, String target) {
    // p(source, target) = (# times source and target appeared together) / (count of source * count of target)
    return sourceTargetCounts.getCount(source, target) / (sourceWordCounts.get(source) * targetWordCounts.get(target));
  }

  public void train(List<SentencePair> trainingPairs) {
    totalNumPairs = trainingPairs.size();

    for (SentencePair pair : trainingPairs) {
      for (String sourceWord : pair.getSourceWords()) {
        for (String targetWord : pair.getTargetWords()) {
            process(sourceWord, targetWord);
        }
      }
    }
  }

  private void process(String source, String target) {
    int sourceCount = (sourceWordCounts.containsKey(source)) ? sourceWordCounts.get(source) + 1 : 1;
    sourceWordCounts.put(source, sourceCount);

    int targetCount = (targetWordCounts.containsKey(target)) ? targetWordCounts.get(target) + 1 : 1;
    targetWordCounts.put(target, targetCount);

    sourceTargetCounts.incrementCount(source, target, 1);
  }
}
