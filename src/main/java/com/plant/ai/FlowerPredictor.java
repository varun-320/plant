package com.plant.ai;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import java.io.File;
import java.util.List;
import java.util.Arrays;

public class FlowerPredictor {
    public static void main(String[] args) throws Exception {
        // 1. Load the "Knowledge" we just trained
        File modelFile = new File("E:/plant/flowersystem/flower_model.zip");
        if (!modelFile.exists()) {
            System.out.println("Model file not found! Wait for training to finish.");
            return;
        }
        MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, true);

        // 2. Load the image you want to test (Change this path to your test image!)
        File testImage = new File("E:/plant/test_flower.jpg");
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(testImage);

        // 3. Normalize the test image (Must be same as training!)
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);

        // 4. Ask the AI for a prediction
        INDArray output = model.output(image);
        
        // 5. Match the result to your flower labels
        // Note: Labels are usually alphabetical based on your folder names
        // Example: [0: Daisy, 1: Rose, 2: Sunflower...]
        System.out.println("AI Prediction Probabilities: " + output.toString());
        
        // Find the index with the highest probability
        double max = -1;
        int bestGuess = -1;
        for (int i = 0; i < output.columns(); i++) {
            if (output.getDouble(i) > max) {
                max = output.getDouble(i);
                bestGuess = i;
            }
        }
        System.out.println("The AI thinks this is flower type #" + bestGuess + " with " + (max * 100) + "% confidence.");
    }
}