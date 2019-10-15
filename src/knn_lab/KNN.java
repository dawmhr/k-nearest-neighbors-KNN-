/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package knn_lab;

import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 *
 * @author bis671
 */
public class KNN {

    String traningDataFile;
    String predictDataFile;
    double besK = 0;
    String type;

    public KNN(String traningDataFile, String predictDataFile, String type) {
        this.traningDataFile = traningDataFile;
        this.predictDataFile = predictDataFile;
        this.type = type;
    }

    public Instances getDataSet(String filename) {
        ArffLoader loader = new ArffLoader();
        try {
            loader.setFile(new File(filename));
            Instances dataSet = loader.getDataSet();
            return dataSet;
        } catch (IOException ex) {
            Logger.getLogger(KNN.class.getName()).log(Level.SEVERE, null, ex);
        }

        return null;
    }

    public void train() {
        Instances trainData = getDataSet(traningDataFile);
        trainData.setClassIndex(trainData.numAttributes() - 1);
        Classifier ibk = new IBk();
        double percentComparison = 0;
        double currentPercentComparision = 0;

        try {
            for (int i = 3; i < 20; i += 2) {
                ibk = new IBk(i);
                ibk.buildClassifier(trainData);
                System.out.println("ibk ---->" + ibk);
                Evaluation eval = new Evaluation(trainData);
                eval.evaluateModel(ibk, trainData);
                currentPercentComparision = eval.correct() / eval.incorrect();
                if (currentPercentComparision > percentComparison) {
                    percentComparison = currentPercentComparision;
                    besK = i;
                }
                System.out.println(eval.toSummaryString());
                System.out.println(eval.toClassDetailsString());
                System.out.println(eval.toMatrixString());
            }
            System.out.println("besK--> " + besK);
        } catch (Exception ex) {
            Logger.getLogger(KNN.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    public void predict() {

        try {
            Classifier ibk = new IBk((int) besK);
            Instances trainData = getDataSet(traningDataFile);
            trainData.setClassIndex(trainData.numAttributes() - 1);
            ibk.buildClassifier(trainData);
            System.out.println("Prediction");
            Instances predictData = getDataSet(predictDataFile);
            predictData.setClassIndex(predictData.numAttributes() - 1);
            Instance predictInstance;
            double answer;
            for (int i = 0; i < predictData.numInstances(); i++) {
                predictInstance = predictData.instance(i);
                answer = ibk.classifyInstance(predictInstance);
                System.out.println(answer);
                if (type == "iris") {
                    System.out.println(answer == 0 ? "Iris-setosa"
                            : answer == 1 ? "Iris-versicolor"
                                    : answer == 2 ? "Iris-virginica" : "");
                }

                if (type == "car") {
                    System.out.println(answer == 0 ? "unacc"
                            : answer == 1 ? "acc"
                                    : answer == 2 ? "good" : "vgood");
                }

            }

        } catch (Exception ex) {
            Logger.getLogger(KNN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}
