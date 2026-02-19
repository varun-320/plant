package com.plant.ai;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import java.io.File;
import java.util.Arrays;
import java.util.List;

public class FlowerPredictor {
    // List your folder names in ALPHABETICAL order here
    private static final List<String> LABELS = Arrays.asList("Daisy", "Lotus", "Rose", "Sunflower", "Tulip");

    public static void main(String[] args) throws Exception {
        File modelFile = new File("E:/plant/flowersystem/flower_model.zip");
        if (!modelFile.exists()) {
            System.out.println("Model file not found! Please check the path.");
            return;
        }

        System.out.println("Loading trained model...");
        MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, true);

        // Load and prepare the test image
        File testImage = new File("E:/plant/new.jpg");
        if (!testImage.exists()) {
            System.out.println("Test image not found at E:/plant/test_flower.jpg");
            return;
        }

        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(testImage);

        // Preprocessing (scale pixels to 0-1 range)
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);

        // Get prediction
        INDArray output = model.output(image);
        
        double max = -1;
        int bestGuessIndex = -1;
        for (int i = 0; i < output.columns(); i++) {
            if (output.getDouble(i) > max) {
                max = output.getDouble(i);
                bestGuessIndex = i;
            }
        }

        String flowerName = (bestGuessIndex < LABELS.size()) ? LABELS.get(bestGuessIndex) : "Unknown";

        System.out.println("\n--- AI Results ---");
        System.out.println("Predicted Flower: " + flowerName);
        System.out.println("Confidence: " + String.format("%.2f", max * 100) + "%");
        System.out.println("------------------\n");
    }
}