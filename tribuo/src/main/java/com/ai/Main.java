package com.ai;

import com.ai.tribuo.MlAlgorithms;
import io.github.resilience4j.core.StopWatch;

import java.io.IOException;
import java.util.Date;

public class Main {

    public static void main(String[] args) {

        MlAlgorithms mlAlgorithms = new MlAlgorithms();
        try {
            mlAlgorithms.classification();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println("Task Complete...");
    }
}
