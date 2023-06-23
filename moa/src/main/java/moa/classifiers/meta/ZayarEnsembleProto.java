package moa.classifiers.meta;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ZayarEnsembleProto extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    protected List<Classifier> ensemble;
    protected double[] predictivePerformances;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption windowSizeOption = new IntOption("windowSize", 'w',
            "The length of the window (w).", 1000, 1, Integer.MAX_VALUE);

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of learners in the ensemble (S).", 10, 1, Integer.MAX_VALUE);
    public IntOption seedOption = new IntOption("seed", 'r',
            "Seed for the random number generator (seed).", 1, 1, Integer.MAX_VALUE);

    protected Random classifierRandom;
    protected Classifier candidateModel;

    // Accumulated performance variables
    protected int numInstances;
    protected int numCorrectPredictions;

    public ZayarEnsembleProto() {
        this.ensemble = new ArrayList<>();
        this.predictivePerformances = new double[getEnsembleSize()]; // Set size based on ensemble size option
        this.classifierRandom = new Random(getSeed());
        this.numInstances = 0;
        this.numCorrectPredictions = 0;
    }

    @Override
    public void resetLearningImpl() {
        this.ensemble.clear();
        for (int i = 0; i < getEnsembleSize(); i++) {
            Classifier classifier = ((Classifier) getPreparedClassOption(baseLearnerOption)).copy();

            // Set hyperparameters for base learner
            if (classifier instanceof HoeffdingTree) {
                HoeffdingTree ht = (HoeffdingTree) classifier;

                // Set random hyperparameters for base learner
                int gracePeriod = getRandomValue(10, 200, 10);
                double splitConfidence = getRandomValue(0.0, 1.0, 0.05);
                double tieThreshold = getRandomValue(0.0, 1.0, 0.05);

                ht.gracePeriodOption.setValue(gracePeriod);
                ht.splitConfidenceOption.setValue(splitConfidence);
                ht.tieThresholdOption.setValue(tieThreshold);
            }

            this.ensemble.add(classifier);
        }
        this.predictivePerformances = new double[getEnsembleSize()]; // Reset size based on ensemble size option
        this.classifierRandom = new Random(getSeed());
        this.numInstances = 0;
        this.numCorrectPredictions = 0;
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        if (getWeightSeenByModel() % getWindowSize() == 0) {
            this.candidateModel = ((Classifier) getPreparedClassOption(baseLearnerOption)).copy();
            this.candidateModel.resetLearning();
            this.candidateModel.setRandomSeed(getSeed());
        }

        if (this.ensemble.isEmpty()) {
            this.ensemble.add(this.candidateModel);
            this.predictivePerformances = new double[]{measurePredictivePerformance(instance)};
        } else {
            for (int i = 0; i < this.ensemble.size(); i++) {
                int k = MiscUtils.poisson(1.0, this.classifierRandom);
                if (k > 0) {
                    Classifier model = this.ensemble.get(i);
                    double[] votes = model.getVotesForInstance(instance);
                    double predictedClass = getPredictedClass(votes);
                    double trueClass = instance.classValue();
                    double accuracy = predictedClass == trueClass ? 1.0 : 0.0;
                    this.numCorrectPredictions += accuracy;
                    this.numInstances++;
                    this.predictivePerformances[i] = (double) this.numCorrectPredictions / this.numInstances;
                    model.trainOnInstance(instance);
                    double basePerformance = measurePredictivePerformance(instance);
                    if (basePerformance > this.predictivePerformances[i]) {
                        this.ensemble.set(i, model.copy());
                        this.predictivePerformances[i] = basePerformance;
                    } else {
                        // IGNORE base learner
                    }
                }
            }

            if (getWeightSeenByModel() % getWindowSize() == 0) {
                double candidatePerformance = measurePredictivePerformance(instance);
                int leastAccurateModelIndex = findLeastAccurateModel();
                if (candidatePerformance > this.predictivePerformances[leastAccurateModelIndex]) {
                    this.ensemble.set(leastAccurateModelIndex, this.candidateModel);
                    this.predictivePerformances[leastAccurateModelIndex] = candidatePerformance;
                } else {
                    // IGNORE c
                }
            }
        }
    }

    private double measurePredictivePerformance(Instance instance) {
        double[] votes = getVotesForInstance(instance);
        double predictedClass = getPredictedClass(votes);
        double trueClass = instance.classValue();
        double accuracy = predictedClass == trueClass ? 1.0 : 0.0;
        this.numCorrectPredictions += accuracy;
        this.numInstances++;
        return (double) this.numCorrectPredictions / this.numInstances;
    }

    private int findLeastAccurateModel() {
        int leastAccurateModelIndex = 0;
        double minPerformance = this.predictivePerformances[0];
        for (int i = 1; i < this.predictivePerformances.length; i++) {
            if (this.predictivePerformances[i] < minPerformance) {
                leastAccurateModelIndex = i;
                minPerformance = this.predictivePerformances[i];
            }
        }
        return leastAccurateModelIndex;
    }

    private double getPredictedClass(double[] votes) {
        return MiscUtils.maxIndex(votes);
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        DoubleVector combinedVote = new DoubleVector();
        for (Classifier model : this.ensemble) {
            double[] modelVotes = model.getVotesForInstance(instance);
            double modelPerformance = model.trainingWeightSeenByModel() > 0 ? this.predictivePerformances[this.ensemble.indexOf(model)] : 0.0;
            addWeightedValues(combinedVote, modelVotes, modelPerformance);
        }
        return combinedVote.getArrayRef();
    }

    private void addWeightedValues(DoubleVector combinedVote, double[] values, double weight) {
        for (int i = 0; i < values.length; i++) {
            combinedVote.setValue(i, combinedVote.getValue(i) + values[i] * weight);
        }
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public String getPurposeString() {
        return "ZayarEnsemble is an ensemble classifier that combines the predictions of multiple base classifiers. " +
                "It uses an incremental on-line bagging approach to dynamically update the ensemble based on the performance of individual models. " +
                "The ensemble size and base learner can be configured to suit the problem at hand. " +
                "ZayarEnsemble aims to provide accurate and robust predictions for multi-class classification tasks.";
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size", this.ensemble != null ? this.ensemble.size() : 0)};
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        String indentStr = "";
        for (int i = 0; i < indent; i++) {
            indentStr += "  ";
        }
        out.append(indentStr).append("ZayarEnsemble Classifier\n");
        out.append(indentStr).append("------------------------------\n");
        out.append(indentStr).append("ZayarEnsemble is an ensemble classifier that combines the predictions of multiple base classifiers. It utilizes an incremental on-line bagging approach to dynamically update the ensemble based on the performance of individual models.\n");
        out.append(indentStr).append("Key Features:\n");
        out.append(indentStr).append("  - Customizable ensemble size (default: 10)\n");
        out.append(indentStr).append("  - Configurable base learner (default: trees.HoeffdingTree)\n");
        out.append(indentStr).append("  - Variable window size (default: 1000)\n");
        out.append(indentStr).append("  - Random seed for the number generator (default: 1)\n");
        out.append(indentStr).append("Purpose:\n");
        out.append(indentStr).append("ZayarEnsemble aims to provide accurate and robust predictions for multi-class classification tasks. By leveraging the strengths of multiple base classifiers, it enhances the predictive power and improves the overall performance of the classification process.\n");
    }

    protected int getWindowSize() {
        return windowSizeOption.getValue();
    }

    protected int getEnsembleSize() {
        return ensembleSizeOption.getValue();
    }

    protected int getSeed() {
        return seedOption.getValue();
    }

    protected double getWeightSeenByModel() {
        double weight = 0.0;
        for (Classifier model : this.ensemble) {
            weight += model.trainingWeightSeenByModel();
        }
        return weight;
    }

    private int getRandomValue(int minValue, int maxValue, int step) {
        int range = (maxValue - minValue) / step + 1;
        int randomIndex = classifierRandom.nextInt(range);
        return minValue + randomIndex * step;
    }

    private double getRandomValue(double minValue, double maxValue, double step) {
        int range = (int) ((maxValue - minValue) / step) + 1;
        int randomIndex = classifierRandom.nextInt(range);
        return minValue + randomIndex * step;
    }

}
