package com.plant.ai;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class FlowerModel {
    
    public static MultiLayerConfiguration getConfiguration(int numClasses) {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001)) // The speed at which the AI learns
            .list()
            // Layer 1: Look for basic edges/shapes
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nIn(3) // RGB Channels
                .stride(1, 1)
                .nOut(20)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build())
            // Layer 2: Shrink the data to focus on important parts
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            // Layer 3: Connect all patterns to a decision
            .layer(new DenseLayer.Builder().activation(Activation.RELU)
                .nOut(200).build())
            // Layer 4: Final Output (one for each flower type)
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(numClasses)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutionalFlat(224, 224, 3))
            .build();
    }
}