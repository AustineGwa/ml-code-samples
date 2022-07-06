package com.ai.tribuo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Paths;

import org.tribuo.*;
import org.tribuo.clustering.evaluation.ClusteringEvaluator;
import org.tribuo.clustering.example.GaussianClusterDataSource;
import org.tribuo.clustering.kmeans.KMeansTrainer;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.classification.*;
import org.tribuo.classification.evaluation.*;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import com.fasterxml.jackson.databind.*;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.config.json.*;
import org.tribuo.*;
import org.tribuo.util.Util;
import org.tribuo.clustering.*;
import org.tribuo.clustering.evaluation.*;
import org.tribuo.clustering.example.GaussianClusterDataSource;
import org.tribuo.clustering.kmeans.*;
import org.tribuo.clustering.kmeans.KMeansTrainer.Distance;
import org.tribuo.clustering.kmeans.KMeansTrainer.Initialisation;


public class TribuoMlAlgorithms {


    /*
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

    /*
    For Explanations on this algorithm
    https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
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


    /*
     For Explanations on this algorithm
     https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
     */
    public void kMeansClustering() throws Exception{
         /*
        Loading data
        We're going to sample two datasets (using different seeds) one for fitting the cluster centroids, and one to measure clustering performance.

        The defaults for the data source are:
            N([ 0.0,0.0], [[1.0,0.0],[0.0,1.0]])
            N([ 5.0,5.0], [[1.0,0.0],[0.0,1.0]])
            N([ 2.5,2.5], [[1.0,0.5],[0.5,1.0]])
            N([10.0,0.0], [[0.1,0.0],[0.0,0.1]])
            N([-1.0,0.0], [[1.0,0.0],[0.0,0.1]])
        */
        var eval = new ClusteringEvaluator();
        var data = new MutableDataset<>(new GaussianClusterDataSource(500, 1L));
        var test = new MutableDataset<>(new GaussianClusterDataSource(500, 2L));

        /*
        Model Training
        We'll first fit a K-Means using 5 centroids, a maximum of 10 iterations, using the euclidean distance and a single computation thread.
         */
        var trainer = new KMeansTrainer(5, /* centroids */10, /* iterations */
                Distance.EUCLIDEAN, /* distance function */
                1, /* number of compute threads */
                1 /* RNG seed */
        );
        var startTime = System.currentTimeMillis();
        var model = trainer.train(data);
        var endTime = System.currentTimeMillis();
        System.out.println("Training with 5 clusters took " + Util.formatDuration(startTime,endTime));
        var centroids = model.getCentroids();
        for (var centroid : centroids) {
            System.out.println(centroid);
        }

        /*
        K-Means++¶
        The training time isn't much different in this case,
         but the K-Means++ initialisation does take longer than the default on larger datasets.
        However the resulting clusters are usually better.
         */

//        var plusplusTrainer = new KMeansTrainer(5,10,Distance.EUCLIDEAN,Initialisation.PLUSPLUS,1,1);
//        var startTime = System.currentTimeMillis();
//        var plusplusModel = plusplusTrainer.train(data);
//        var endTime = System.currentTimeMillis();
//        System.out.println("Training with 5 clusters took " + Util.formatDuration(startTime,endTime));
//        var ppCentroids = plusplusModel.getCentroids();
//        for (var centroid : ppCentroids) {
//            System.out.println(centroid);
//        }

        /*
         Model evaluation
         Tribuo uses the normalized mutual information to measure the quality of two clusterings.
         This avoids the issue that swapping the id number of any given centroid doesn't change the overall clustering.
         We're going to compare against the ground truth cluster labels from the data generator.

         */
//        First for the training data:
        var trainEvaluation = eval.evaluate(model,data);
        System.out.println(" Evaluation training data \n" + trainEvaluation.toString());

//        Then for the unseen test data:
        var testEvaluation = eval.evaluate(model,test);
        System.out.println(" Evaluation unseen test data\n" +testEvaluation.toString());




    }
}
