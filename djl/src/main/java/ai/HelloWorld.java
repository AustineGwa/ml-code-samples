package ai;

import ai.djl.Application;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;

import java.io.IOException;
import java.nio.file.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ai.djl.*;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.ndarray.types.*;
import ai.djl.training.*;
import ai.djl.training.dataset.*;
import ai.djl.training.initializer.*;
import ai.djl.training.loss.*;
import ai.djl.training.listener.*;
import ai.djl.training.evaluator.*;
import ai.djl.training.optimizer.*;
import ai.djl.training.util.*;
import ai.djl.basicmodelzoo.cv.classification.*;
import ai.djl.basicmodelzoo.basic.*;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class HelloWorld {

    private void trainAndInferWithComments() throws IOException, TranslateException, MalformedModelException {
        /*
         Creating A neural network

         The first thing to figure out when trying to build a neural network,is knowing what your function signature is.
         What are your input types and output types? Because most models use relatively consistent signatures, we refer to them as Applications.
         Within the Applications interface, you can find a list of some of the more common model applications used in deep learning.
         */
        Application application = Application.CV.IMAGE_CLASSIFICATION;

        /*
        Dataset MNIST

        algorithm Multilayer Perceptron (simplest and oldest deep learning networks)
        The MLP is organized into layers.

        The first layer is the input layer which contains your input data and the last layer is the output layer which produces the final result of the network.
        Between them are layers referred to as hidden layers.
        Having more hidden layers and larger hidden layers allows the MLP to represent more complex functions.

        Between each pair of layers is a linear operation
        (sometimes called a FullyConnected operation because each number in the input is connected to each number in the output by a matrix multiplication).
        Not pictured, there is also a non-linear activation function after each linear operation.
        */

        /*
        NDArray

        The core data type used for working with deep learning is the NDArray. An NDArray represents a multidimensional,fixed-size homogeneous array.
        It has very similar behavior to the Numpy python package with the addition of efficient computing. We also have a helper class,
        the NDList which is a list of NDArrays which can have different sizes and data types.

        */

        long inputSize = 28*28;
        long outputSize = 10;

        /*
        Block API
        In DJL,Blocks serve a purpose similar to functions that convert an input NDList to an output NDList.
        They can represent single operations, parts of a neural network, and even the whole neural network.
        What makes blocks special is that they contain a number of parameters that are used in their function and are trained during deep learning.
        As these parameters are trained, the function represented by the blocks get more and more accurate.
        */

        SequentialBlock block = new SequentialBlock();

        /*
        Add blocks to SequentialBlock

        An MLP is organized into several layers. Each layer is composed of a Linear Block and a non-linear activation function.
        If we just had two linear blocks in a row, it would be the same as a combined linear block ($f(x) = W_2(W_1x) = (W_2W_1)x = W_{combined}x$).
        An activation is used to intersperse between the linear blocks to allow them to represent non-linear functions.
        We will use the popular ReLU as our activation function.

        The first layer and last layers have fixed sizes depending on your desired input and output size.
        However, you are free to choose the number and sizes of the middle layers in the network.
        We will create a smaller MLP with two middle layers that gradually decrease the size.
        Typically, you would experiment with different values to see what works the best on your data set.
         */
        block.add(Blocks.batchFlattenBlock(inputSize));
        block.add(Linear.builder().setUnits(128).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(64).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(outputSize).build());

        /*
        Training the model
         */

        /*
        Prepare MNIST dataset for training

        In order to train, you must create a Dataset class to contain your training data.
        A dataset is a collection of sample input/output pairs for the function represented by your neural network.
        Each single input/output is represented by a Record.Each record could have multiple arrays of inputs or outputs
         such as an image question and answer dataset where the input is both an image and a question about the image while the output is the answer to the question.

        Because data learning is highly parallelizable, training is often done not with a single record at a time, but a Batch.
        This can lead to significant performance gains, especially when working with images

         */

        int batchSize = 32;
        Mnist mnist = Mnist.builder().setSampling(batchSize, true).build();
        mnist.prepare(new ProgressBar());

        /*
        Create your Model

        Next we will build a model. A Model contains a neural network Block along with additional artifacts used for the training process. It possesses additional information about the inputs, outputs, shapes, and data types you will use. Generally, you will use the Model once you have fully completed your Block.

        In this part of the tutorial, we will use the built-in Multilayer Perceptron Block from the Model Zoo. To learn how to build it from scratch, see the previous tutorial: Create Your First Network.

        Because images in the MNIST dataset are 28x28 grayscale images, we will create an MLP block with 28 x 28 input. The output will be 10 because there are 10 possible classes (0 to 9) each image could be. For the hidden layers, we have chosen new int[] {128, 64} by experimenting with different values.

         */

        Model model = Model.newInstance("mlp");
        model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));

        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                //softmaxCrossEntropyLoss is a standard loss for classification problems
                .addEvaluator(new Accuracy()) // Use accuracy so we humans can understand how accurate the model is
                .addTrainingListeners(TrainingListener.Defaults.logging());

        // Now that we have our training configuration, we should create a new trainer for our model
        Trainer trainer = model.newTrainer(config);
        trainer.initialize(new Shape(1, 28 * 28));
        // Deep learning is typically trained in epochs where each epoch trains the model on each item in the dataset once.
        int epoch = 2;
        EasyTrain.fit(trainer, epoch, mnist, null);



        /*
        Save your model
        Once your model is trained, you should save it so that it can be reloaded later.
        You can also add metadata to it such as training accuracy, number of epochs trained, etc that can be used when loading the model or when examining it.

         */

        Path modelDir = Paths.get("build/mlp");
        Files.createDirectories(modelDir);
        model.setProperty("Epoch", String.valueOf(epoch));
        model.save(modelDir, "mlp");

        /*
        Inference with your model
        Load your handwritten digit image
        Load your model
         */
        var img = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/0.png");
        img.getWrappedImage();

        Path modelDir2 = Paths.get("build/mlp");
        Model model2 = Model.newInstance("mlp");
        model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));
        model.load(modelDir2);

        /*
        Create a Translator

        The Translator is used to encapsulate the pre-processing and post-processing functionality of your application.
        The input to the processInput and processOutput should be single data items, not batches.
         */

        Translator<Image, Classifications> translator = new Translator<Image, Classifications>() {

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                // Convert Image to NDArray
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
                return new NDList(NDImageUtils.toTensor(array));
            }

            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                // Create a Classifications with the output probabilities
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }

            @Override
            public Batchifier getBatchifier() {
                // The Batchifier describes how to combine a batch together
                // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
                return Batchifier.STACK;
            }
        };

        /*
        Create Predictor

        Using the translator, we will create a new Predictor. The predictor is the main class to orchestrate the inference process.
        During inference, a trained model is used to predict values, often for production use cases.
        The predictor is NOT thread-safe, so if you want to do prediction in parallel, you should call newPredictor multiple times to create a predictor object for each thread.

        */

        var predictor = model.newPredictor(translator);

        /*
         run inference

         With our predictor, we can simply call the predict method to run inference.
         For better performance, you can also call batchPredict with a list of input items.
         Afterwards, the same predictor should be used for further inference calls.
         */

        var classifications = predictor.predict(img);
    }

    private static void train() throws IOException, TranslateException {

        Application application = Application.CV.IMAGE_CLASSIFICATION;
        long inputSize = 28*28;
        long outputSize = 10;
        SequentialBlock block = new SequentialBlock();
        block.add(Blocks.batchFlattenBlock(inputSize));
        block.add(Linear.builder().setUnits(128).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(64).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(outputSize).build());
        int batchSize = 32;
        Mnist mnist = Mnist.builder().setSampling(batchSize, true).build();
        mnist.prepare(new ProgressBar());
        Model model = Model.newInstance("mlp");
        model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                //softmaxCrossEntropyLoss is a standard loss for classification problems
                .addEvaluator(new Accuracy()) // Use accuracy so we humans can understand how accurate the model is
                .addTrainingListeners(TrainingListener.Defaults.logging());

        // Now that we have our training configuration, we should create a new trainer for our model
        Trainer trainer = model.newTrainer(config);
        trainer.initialize(new Shape(1, 28 * 28));
        // Deep learning is typically trained in epochs where each epoch trains the model on each item in the dataset once.
        int epoch = 2;
        EasyTrain.fit(trainer, epoch, mnist, null);

        Path modelDir = Paths.get("build/mlp");
        Files.createDirectories(modelDir);
        model.setProperty("Epoch", String.valueOf(epoch));
        model.save(modelDir, "mlp");
    }

    private static void infer() throws IOException, MalformedModelException, TranslateException {
        var img = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/0.png");
        img.getWrappedImage();
        Path modelDir2 = Paths.get("build/mlp");
        Model model = Model.newInstance("mlp");
        model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));
        model.load(modelDir2);
        Translator<Image, Classifications> translator = new Translator<>() {

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                // Convert Image to NDArray
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
                return new NDList(NDImageUtils.toTensor(array));
            }

            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                // Create a Classifications with the output probabilities
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }

            @Override
            public Batchifier getBatchifier() {
                // The Batchifier describes how to combine a batch together
                // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
                return Batchifier.STACK;
            }
        };
        var predictor = model.newPredictor(translator);
        var classifications = predictor.predict(img);
        System.out.println(classifications);
    }

    public static void main(String[] args) {

        try{
//            train();
//            infer();
        }catch (Exception exception){
            System.out.println(exception.getMessage());
        }

    }
}
