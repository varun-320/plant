package com.plant.ai;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class FlowerApplication {

    public static void main(String[] args) {
        // This is the line Java is looking for!
        SpringApplication.run(FlowerApplication.class, args);
    }
}