package com.ai;

import com.ai.tribuo.TribuoMlAlgorithms;

import java.io.IOException;

public class Main {

    public static void main(String[] args) {

        TribuoMlAlgorithms tribuoMlAlgorithms = new TribuoMlAlgorithms();
        try {
            tribuoMlAlgorithms.logisticRegression();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
