package edu.stanford.nlp.mt.decoder.feat;

import java.util.List;

import edu.stanford.nlp.mt.util.FeatureValue;
import edu.stanford.nlp.mt.util.Featurizable;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.decoder.feat.RuleFeaturizer;
import edu.stanford.nlp.util.Generics;

/**
 * A rule featurizer.
 */
public class MyFeaturizer implements RuleFeaturizer<IString, String> {

  @Override
  public void initialize() {
    // Do any setup here.
  }

  @Override
  public List<FeatureValue<String>> ruleFeaturize(
      Featurizable<IString, String> f) {

    // TODO: Return a list of features for the rule. Replace these lines
    // with your own feature.
    List<FeatureValue<String>> features = Generics.newLinkedList();
    final String regex = "([\\d]+|[^\\s\\w])";
    final String regex1 = "[\\d]+";

    int srcNumDigits = 0;
    for (IString srcIString : f.sourcePhrase) {
      if (srcIString.toString().matches(regex1)) {
        srcNumDigits++;
      }
    }

    int targetNumDigits = 0;
    for (IString targetIString : f.targetPhrase) {
      if (targetIString.toString().matches(regex1)) {
        targetNumDigits++;
      }
    }

    double finalValue = (srcNumDigits == 0) ? 0 : (double) targetNumDigits / srcNumDigits;

    features.add(new FeatureValue<String>("MyFeature", finalValue));
    return features;
  }

  @Override
  public boolean isolationScoreOnly() {
    return false;
  }
}
