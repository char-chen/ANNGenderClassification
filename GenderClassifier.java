import java.io.*;
import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;
import java.nio.charset.Charset;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/*
 Provide relative path of the name of directory that has subdirectories containing male and female images.
 For example if the directory name is Train --- java GenderClassifier -train Train/
 Train/ must have Male/ and Female/
*/

public class GenderClassifier {
    
    public static void main(String[] args) {
        ArrayList<Matrix> data = new ArrayList<Matrix>();

        if (args[0].equals("-train")) {
            loadData(data, args[1] + "Male/", 1);
            loadData(data, args[1] + "Female/", 0);
            System.out.println("Training Neural Network...");
            System.out.println("Epoch\tTraining Accuracy");
            train(data);
            System.out.println();
            System.out.println("Training Neural Network with 5-fold Cross Validation 10 times...");
            System.out.println("Epoch\tFold\tTraining Mean\t\tTraining Stdev\t\tTest Error\t\tTest Stdev");
            trainCV(data, 5);
        }
        else if (args[0].equals("-test")) {
            loadData(data, args[1], -1);
            System.out.println("Testing Neural Network...");
            test(data);
        }
        else
            System.out.println("Error: Invalid Arguments");
    }
    
    private static void train(ArrayList<Matrix> data) {
        NeuralNetwork NN = new NeuralNetwork();
        Collections.shuffle(data);
        double error = 0, prev_error = 0;
        for (int i = 0; Math.abs(prev_error - error) > 0.0005 || i == 0; i++) {
            prev_error = error;
            double correct = 0;
            for(Matrix j : data) {
                NN.feed_forward(j);
                NN.back_propagate();
                int predict = NN.predict() ? 1 : 0;         
                if (predict == j.gender)
                    correct++;
            }
            error = 1 - (correct / data.size());
            double accuracy = correct / data.size();
            System.out.println(i + "\t" + accuracy);
        }

        try {
            FileOutputStream fileOut = new FileOutputStream("weights.txt");
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(NN);
            out.close();
        } catch (IOException i) {
            i.printStackTrace();
            System.exit(7);
        }
    }
    
    public static void trainCV(ArrayList<Matrix> data, int k) {
        int partition_size = data.size() / k;
        int test_size = partition_size + data.size() % k;
        
        for (int i = 0; i < 10; i++) {
            Collections.shuffle(data);
            double training_error = 0;
            double validation_error = 0;
            NeuralNetwork NN = new NeuralNetwork(); 
            
            for (int fold = 1; fold <= k; fold++) {
                ArrayList<Double> training_accuracy = new ArrayList<Double>();
                ArrayList<Double> testing_accuracy = new ArrayList<Double>();
                ArrayList<Matrix> training_set = new ArrayList<Matrix>();
                ArrayList<Matrix> testing_set = new ArrayList<Matrix>();
                for (int j = 0; j < data.size(); j++) {
                    if (j >= (fold - 1) * partition_size && j < (fold - 1) * partition_size + test_size - 1)
                        testing_set.add(data.get(j));
                    else
                        training_set.add(data.get(j));
                }
                
                double correct = 0;
                for (Matrix face : training_set) {
                    NN.feed_forward(face);
                    NN.back_propagate();
                    int predict = NN.predict() ? 1 : 0;         
                    if (predict == face.gender)
                        correct++;
                }
                training_accuracy.add(correct / training_set.size());
                
                correct = 0;
                for (Matrix face : testing_set) {
                    NN.test(face);
                    int predict = NN.predict() ? 1 : 0;         
                    if (predict == face.gender)
                        correct++;
                }
                testing_accuracy.add(correct / testing_set.size());
                
                double training_sum = 0;
                for (double t : training_accuracy)
                    training_sum += t;
                double training_mean = training_sum / training_accuracy.size();
                
                double training_stsum = 0;
                for (double t : training_accuracy) {
                    double diff = t - training_mean;
                    training_stsum += Math.pow(diff, 2);
                }
                double training_std = Math.pow(training_stsum/training_accuracy.size(), 0.5);
                
                double testing_sum = 0;
                for (double t : testing_accuracy)
                    testing_sum += t;
                double testing_mean = testing_sum / testing_accuracy.size();

                double testing_stsum = 0;
                for (double t : testing_accuracy)
                    testing_stsum += Math.pow(t - testing_mean, 2);
                double testing_std = Math.pow(testing_stsum/testing_accuracy.size(), 0.5);
                System.out.println(i + 1 + "\t" + fold + "\t" + training_mean + "\t" + training_std + "\t\t\t" + testing_mean + "\t" + testing_std);  
            }
        }
    }
    
    public static void loadData(ArrayList<Matrix> set, String directory, int gender) {
        for (String image : (new File(directory)).list()) {
            try {
                FileInputStream stream = new FileInputStream(new File(directory + "/" + image));
                FileChannel fc = stream.getChannel();
                MappedByteBuffer buffer = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
                set.add(new Matrix(directory + "/" + image, Charset.defaultCharset().decode(buffer).toString(), gender));
                stream.close();
            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
            }
        }
    }
    
    private static void test(ArrayList<Matrix> data) {
        System.out.println("\ti\tname\t\tgender\terror\tconfidence");
        NeuralNetwork NN = null;
        try {   
            FileInputStream file_in = new FileInputStream("weights.txt");
            ObjectInputStream in = new ObjectInputStream(file_in);
            NN = (NeuralNetwork)in.readObject();
            in.close();
            file_in.close();
        } catch (IOException i) {
            i.printStackTrace();
            System.exit(5);
        } catch (ClassNotFoundException c) {
            System.out.println("Class not found");
            c.printStackTrace();
            System.exit(6);
        }
        for (int i = 0; i < data.size(); i++) {
            NeuralNetwork.OutputUnit best = NN.test(data.get(i));
            double confidence = Math.pow(best.target - best.output, 2);
            System.out.println("\t"+ i +"\t" + data.get(i).name + "\t" + best.gender + "\t" + NN.predict() + "\t"+ confidence);
        }
    }
}

class NeuralNetwork implements Serializable {
    int input_layer_size = Matrix.length * Matrix.width;
    int hidden_layer_size = 15;
    int output_layer_size = 2;
    int num_hidden_layers = 1;
    double learning_rate = 0.3;
    
    InputUnit[] input_layer;
    HiddenUnit[][] hidden_layers;
    OutputUnit[] output_layer;
   
    class Neuron implements Serializable {
        public double input;
        public double output;
        public double error;
    }
     
    public class InputUnit implements Serializable {
        public double value;
        public double[] weights;
        public InputUnit(double[] w) {
            weights = w;
            weights[0] = 1;
        }
    }
      
    public class HiddenUnit extends Neuron implements Serializable {
        public double[] weights;
        public HiddenUnit(double[] w) {
            weights = w;
            weights[0] = 1;
        }
    }
    
    public class OutputUnit extends Neuron implements Serializable {            
        public double target;
        public int gender;
        public OutputUnit(int g) {
            gender = g;
        }
    }
    
    public NeuralNetwork() {
        Random rand = new Random();
        input_layer = new InputUnit[input_layer_size];
        hidden_layers = new HiddenUnit[num_hidden_layers][hidden_layer_size];
        output_layer = new OutputUnit[output_layer_size];
        output_layer[0] = new OutputUnit(1);
        output_layer[1] = new OutputUnit(0);
        
        for (int i = 0; i < input_layer_size; i++) {
            input_layer[i] = new InputUnit(new double[hidden_layer_size]);
            for (int j = 0; j < hidden_layer_size; j++)
                input_layer[i].weights[j] = rand.nextDouble() - 0.5;
        }
        
        for (int i = 0; i < num_hidden_layers; i++) {
            for (int j = 0; j < hidden_layer_size; j++) {
                if(i != num_hidden_layers - 1) {
                    hidden_layers[i][j] = new HiddenUnit(new double[hidden_layer_size]);
                    for(int k = 0; k < hidden_layer_size; k++) {
                        hidden_layers[i][j].weights[k] = rand.nextDouble() - 0.5;
                    }
                } else {
                    hidden_layers[i][j] = new HiddenUnit(new double[output_layer_size]);
                    for(int k = 0; k < output_layer_size; k++) {
                        hidden_layers[i][j].weights[k] = rand.nextDouble() - 0.5;
                    }
                }
            }
        }
    }
    
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    public void feed_forward(Matrix face) {
        for(int i = 0; i < input_layer_size; i++)
            input_layer[i].value = face.pixels[i];
        
        for(int i = 0; i < num_hidden_layers; i++) {
            for(int j = 0; j < hidden_layer_size; j++) {
                double total = 0.0;
                if(i > 0) {
                    for(int k = 0; k < hidden_layer_size; k++)
                        total += hidden_layers[i - 1][k].output * hidden_layers[i - 1][k].weights[j];
                } else {
                    for(int k = 0; k < input_layer_size; k++)
                        total += input_layer[k].value * input_layer[k].weights[j];
                }
                hidden_layers[i][j].input = total;
                hidden_layers[i][j].output = sigmoid(total);
            }
        }
        
        for(int i = 0; i < output_layer_size; i++) {
            double total = 0;
            for(int j = 0; j < hidden_layer_size; j++)
                total += hidden_layers[num_hidden_layers - 1][j].output * hidden_layers[num_hidden_layers - 1][j].weights[i];
            
            output_layer[i].input = total;
            output_layer[i].output = sigmoid(total);
            output_layer[i].target = output_layer[i].gender == face.gender ? 1.0 : 0.0;
            output_layer[i].error = (output_layer[i].target - output_layer[i].output) * output_layer[i].output * (1 - output_layer[i].output);
        }
    }
    
    public void back_propagate() {
        for(int i = num_hidden_layers - 1; i > -1; i--) {
            for(int j = 0; j < hidden_layer_size; j++) {
                double total = 0;
                if(i != num_hidden_layers - 1) {
                    for(int k = 0; k < hidden_layer_size; k++) {
                        total += hidden_layers[i][j].weights[k] * hidden_layers[i + 1][k].error;
                    }
                } else {
                    for(int k = 0; k < output_layer_size; k++) {
                        total += hidden_layers[i][j].weights[k] * output_layer[k].error;
                    }
                }
                hidden_layers[i][j].error = total;
            }
        }
        
        for(int i = 0; i < hidden_layer_size; i++)
            for(int j = 0; j < input_layer_size; j++)
                input_layer[j].weights[i] += learning_rate * hidden_layers[0][i].error * input_layer[j].value;
        
        for(int i = 0; i < num_hidden_layers - 1; i++)
            for(int j = 0; j < hidden_layer_size; j++)
                for(int k = 0; k < hidden_layer_size; k++)
                    hidden_layers[i][k].weights[j] += learning_rate * hidden_layers[i + 1][j].error * hidden_layers[i][k].output;
        
        for(int i = 0; i < output_layer_size; i++)
            for(int j = 0; j < hidden_layer_size; j++)
                hidden_layers[num_hidden_layers - 1][j].weights[i] += learning_rate * output_layer[i].error * hidden_layers[num_hidden_layers - 1][j].output;
    }
    
    public OutputUnit test(Matrix face) {
        for(int i = 0; i < input_layer_size; i++)
            input_layer[i].value = face.pixels[i];
        
        for(int i = 0; i < num_hidden_layers; i++) {
            for(int j = 0; j < hidden_layer_size; j++) {
                double total = 0.0;
                if(i > 0) {
                    for(int k = 0; k < hidden_layer_size; k++) {
                        total += hidden_layers[i - 1][k].output * hidden_layers[i - 1][k].weights[j];
                    }
                } else {
                    for(int k = 0; k < input_layer_size; k++) {
                        total += input_layer[k].value * input_layer[k].weights[j];
                    }
                }
                hidden_layers[i][j].input = total;
                hidden_layers[i][j].output = sigmoid(total);
            }
        }
        
        OutputUnit best = new OutputUnit(-1);
        best.output = -1;
        
        for(int i = 0; i < output_layer_size; i++) {
            double total = 0.0;
            for(int j = 0; j < hidden_layer_size; j++) 
                total += hidden_layers[num_hidden_layers - 1][j].output * hidden_layers[num_hidden_layers - 1][j].weights[i];
            output_layer[i].input = total;
            output_layer[i].output = sigmoid(total);
            output_layer[i].target = output_layer[i].gender == face.gender ? 1.0 : 0.0;
            output_layer[i].error = (output_layer[i].target - output_layer[i].output) * output_layer[i].output * (1 - output_layer[i].output);
            
            if(output_layer[i].output > best.output)
                best = output_layer[i];
        }
        
        return best;
    }
    
    public boolean predict() {
        return output_layer[0].output > output_layer[1].output;
    }
}

class Matrix {
    public static int length = 128;
    public static int width = 120;
    public int gender;
    public int[] pixels;
    String name;
     
    public Matrix(String n, String raw, int g) {
        pixels = new int[length * width];
        String[] rawpixels = raw.split("[ \n\r]+");
        gender = g;
        name = n;
        for(int i = 0; i < length * width; i++)
            pixels[i] = Integer.valueOf(rawpixels[i]);
    }
    
    public int getPixel(int x, int y) throws Exception {
        if(x > length || y > width || x < 1 || y < 1) {
            throw new Exception("Coordinates out of bounds.");
        }
        
        return pixels[length * (y - 1) + (x - 1)]; 
    }

}
