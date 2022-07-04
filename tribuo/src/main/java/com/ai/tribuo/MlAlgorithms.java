package com.ai.tribuo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Paths;

import org.tribuo.*;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.classification.*;
import org.tribuo.classification.evaluation.*;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import com.fasterxml.jackson.databind.*;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.config.json.*;

public class MlAlgorithms {


    /*
    Setup
    You'll need to get a copy of the irises dataset.
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data

    This method demonstrates Tribuo's classification models to predict Iris species using Fisher's well known Irises dataset
    We'll focus on a simple logistic regression, and investigate the provenance and metadata that Tribuo stores inside each model.

    Logistic regression estimates the probability of an event occurring, such as voted or didn't vote,
    based on a given dataset of independent variables.

    Since the outcome is a probability,the dependent variable is bounded between 0 and 1.
    This method looks at Tribuo's csv loading mechanism, how to train a simple classifier, how to evaluate a classifier on test data,
    what metadata and provenance information is stored inside Tribuo's Model and Evaluation objects,
    and finally how to save and load Tribuo's models.
     */
    public void logisticRegression() throws IOException {

        /*
        The Differences between Linear Regression and Logistic Regression.
        Linear Regression is used to handle regression problems whereas Logistic regression is used to handle the classification problems.
        Linear regression provides a continuous output but Logistic regression provides discreet output.
        */

        /*
        Loading data
        Here we're going to use LabelFactory as we're performing multi-class classification.
        We then pass the labelFactory into the simple CSVLoader which reads all the columns into a DataSource.
        */

        var labelFactory = new LabelFactory();
        var csvLoader = new CSVLoader<>(labelFactory);

        var irisHeaders = new String[]{"sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"};
        var irisesSource = csvLoader.loadDataSource(Paths.get("src/main/resources/datasets/bezdekIris.data"),"species",irisHeaders);
        var irisSplitter = new TrainTestSplitter<>(irisesSource,0.7,1L);
        var trainingDataset = new MutableDataset<>(irisSplitter.getTrain());
        var testingDataset = new MutableDataset<>(irisSplitter.getTest());
        System.out.println(String.format("Training data size = %d, number of features = %d, number of classes = %d",trainingDataset.size(),trainingDataset.getFeatureMap().size(),trainingDataset.getOutputInfo().size()));
        System.out.println(String.format("Testing data size = %d, number of features = %d, number of classes = %d",testingDataset.size(),testingDataset.getFeatureMap().size(),testingDataset.getOutputInfo().size()));

        /*
        Training the model
        A linear model, using a logistic loss, trained with AdaGrad for 5 epochs.
         */
        Trainer<Label> trainer = new LogisticRegressionTrainer();
        System.out.println(trainer.toString());
        Model<Label> irisModel = trainer.train(trainingDataset);

        /*
        Evaluating the model
        it's time to figure out how good the model is
         */
        var evaluator = new LabelEvaluator();
        var evaluation = evaluator.evaluate(irisModel,testingDataset);
        System.out.println(evaluation.toString());
        System.out.println(evaluation.getConfusionMatrix().toString());

        /*
        Model Metadata
         */
        var featureMap = irisModel.getFeatureIDMap();
        for (var v : featureMap) {
            System.out.println(v.toString());
            System.out.println();
        }

        /*
         Model Provenance
         In Tribuo each model tracks it's provenance. It knows how it was created, when it was created, and what data was involved
         The json provenance is verbose, but provides an alternative human readable serialization format.
         */
        var provenance = irisModel.getProvenance();
        System.out.println(ProvenanceUtil.formattedProvenanceString(provenance.getDatasetProvenance().getSourceProvenance()));
        System.out.println(ProvenanceUtil.formattedProvenanceString(provenance.getTrainerProvenance()));
        ObjectMapper objMapper = new ObjectMapper();
        objMapper.registerModule(new JsonProvenanceModule());
        objMapper = objMapper.enable(SerializationFeature.INDENT_OUTPUT);
        String jsonProvenance = objMapper.writeValueAsString(ProvenanceUtil.marshalProvenance(provenance));
        System.out.println(jsonProvenance);

        /*
        Loading and saving models
        Tribuo uses Java Serialization to save and load models.
         */
        File tmpFile = new File("src/main/resources/models/iris-lr-model.ser");
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(tmpFile))) {
            oos.writeObject(irisModel);
        }

    }
}
