/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package knn_lab;

/**
 *
 * @author bis671
 */
public class KNN_Lab {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        String traningFile = "irisTraining.arff";
        String predictDFile = "irisPredict.arff";
        KNN model = new KNN(traningFile, predictDFile, "iris");
        model.train();
        model.predict();

        KNN carmodel = new KNN("carTraining.arff", "carPredict.arff", "car");
        carmodel.train();
        carmodel.predict();
    }

}
