package com.ai;

import com.ai.tribuo.MlAlgorithms;

import java.io.IOException;

public class Main {

    public static void main(String[] args) {

        MlAlgorithms mlAlgorithms = new MlAlgorithms();
        try {
            mlAlgorithms.logisticRegression();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println("Task Complete...");
    }
}
