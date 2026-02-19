package com.plant.ai.controller;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.PostConstruct;
import java.io.File;
import java.util.Arrays;
import java.util.List;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*") // Allows NextJS to connect without security blocks
public class FlowerController {

    private MultiLayerNetwork model;
    // Ensure these match your folder names exactly
    private final List<String> labels = Arrays.asList("Daisy", "Lotus", "Rose", "Sunflower", "Tulip");

    @PostConstruct
    public void init() throws Exception {
        // Loads the brain ONCE when the server starts so it stays in RAM
        File modelFile = new File("E:/plant/flowersystem/flower_model.zip");
        model = MultiLayerNetwork.load(modelFile, true);
        System.out.println(">>> AI Model loaded successfully into the Web Server!");
    }

    @PostMapping("/identify")
    public String identifyFlower(@RequestParam("file") MultipartFile file) {
        try {
            // 1. Receive the file from the browser and save it temporarily
            File tempFile = File.createTempFile("upload", ".jpg");
            file.transferTo(tempFile);

            // 2. Prepare the image (Resize to 224x224, 3 channels)
            NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
            INDArray image = loader.asMatrix(tempFile);
            new ImagePreProcessingScaler(0, 1).transform(image);

            // 3. Run the AI Prediction
            INDArray output = model.output(image);
            int bestGuess = 0;
            double max = -1;

            for (int i = 0; i < output.columns(); i++) {
                if (output.getDouble(i) > max) {
                    max = output.getDouble(i);
                    bestGuess = i;
                }
            }

            // 4. Return the result in JSON format for the website
            return String.format("{\"flower\": \"%s\", \"confidence\": \"%.2f%%\"}", 
                                labels.get(bestGuess), max * 100);

        } catch (Exception e) {
            return "{\"error\": \"" + e.getMessage() + "\"}";
        }
    }
}