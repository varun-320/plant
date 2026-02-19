package com.plant.ai;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.Random;

public class FlowerApp {
    public static void main(String[] args) throws Exception {
        // 1. Load the Data
        File parentDir = new File("E:/plant/flowersystem/data/flowers");
        FileSplit fileSplit = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        ImageRecordReader recordReader = new ImageRecordReader(224, 224, 3, labelMaker);
        recordReader.initialize(fileSplit);
        
        int numClasses = recordReader.getLabels().size();
        System.out.println("Success! Found " + numClasses + " types of flowers.");

        // 2. Prepare Data for the AI (Normalization)
        // Batch size of 16 means the AI looks at 16 flowers at a time
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 16, 1, numClasses);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        // 3. Initialize the Brain (Model)
        MultiLayerConfiguration conf = FlowerModel.getConfiguration(numClasses);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // 4. Start Training
        System.out.println("Starting training... This might take time depending on your CPU.");
        for(int i = 0; i < 10; i++) { // 10 Epochs
            model.fit(dataIter);
            System.out.println("Completed Epoch " + (i + 1));
        }

        // 5. Save the trained model
        File modelFile = new File("E:/plant/flowersystem/flower_model.zip");
        model.save(modelFile, true);
        System.out.println("Model saved successfully at: " + modelFile.getPath());
    }
}