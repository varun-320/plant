package com.plant.ai;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import java.io.File;
import java.util.Random;

public class FlowerApp {
    public static void main(String[] args) throws Exception {
        // Path to your flower images
        File parentDir = new File("E:/plant/flowersystem/data/flowers");
        
        // This tool finds all images (.jpg, .png)
        FileSplit fileSplit = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        
        // This tells the AI to use folder names as labels
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        ImageRecordReader recordReader = new ImageRecordReader(224, 224, 3, labelMaker);
        recordReader.initialize(fileSplit);
        
        System.out.println("Success! Found " + recordReader.getLabels().size() + " types of flowers.");
    }
}