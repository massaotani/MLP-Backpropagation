package mlp_backpropagation;

import java.text.DecimalFormat;

public class Matrix {

    /*Método para transposição de matrizes*/
    public static Double[][] transposta(Double[][] matriz) {
        Double[][] aux = new Double[matriz[0].length][matriz.length];

        for (int i = 0; i < matriz[0].length; i++) {
            for (int j = 0; j < matriz.length; j++) {
                aux[i][j] = matriz[j][i];
            }
        }
        return aux;
    }

    /*Método para povoar uma matriz na forma de matriz linha*/
    public static Double[][] povoaNeuronios(Double[][] entrada, int indice) {
        Double[][] aux = new Double[1][entrada[0].length];
        for (int i = 0; i < entrada[0].length; i++) {
            aux[0][i] = entrada[indice][i];
        }
        return aux;
    }

    //método para setar os valores aleatórios de pesos e bias
    public static Double[][] Randomize(Double[][] m1, int rows, int cols) {
        Double count = 1.0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (count % 2 == 0) {
                    Double d = Math.random();
                    m1[i][j] = Double.valueOf(String.valueOf(d));

                    count -= (j - 1);
                } else {
                    Double d = Math.random();
                    m1[i][j] = Double.valueOf(String.valueOf(d));
                    m1[i][j] *= -1;

                    count += (j + 1);
                }
            }

            count *= (i + 1);
        }

        return m1;
    }

    //Adição de matrizes
    //As 2 matrizes devem ter o mesmo tamanho
    public static Double[][] Add(Double[][] m1, Double[][] m2) {
        Double[][] result = new Double[m1.length][m1[0].length];

        for (int j = 0; j < m2[0].length; j++) {
            for (int k = 0; k < m2.length; k++) {
                result[k][j] = m1[k][j] + m2[k][j];
            }
        }

        return result;
    }

    //Subtração de matrizes
    //As 2 matrizes devem ter o mesmo tamanho
    public static Double[][] subtract(Double[][] m1, Double[][] m2) {
        Double[][] result = new Double[m1.length][m1[0].length];

        for (int j = 0; j < m2[0].length; j++) {
            for (int k = 0; k < m2.length; k++) {
                result[k][j] = m1[k][j] - m2[k][j];
            }
        }

        return result;
    }

    //Produto Hadamard
    //As 2 matrizes devem ter o mesmo tamanho
    /*Consiste na multiplicação de 2 matrizes de tamanho YxZ, para obtenção de uma nova matriz YxZ.
    O calculo é feito de forma diferente da multiplicação tradicional. No Produto Hadamard,
    é necessário calcular o m1[i][j]*m2[i][j] para i=0 até i=Y e j=0 até j=Z
     */
    public static Double[][] ProdutoHadamard(Double[][] m1, Double[][] m2) {
        Double[][] result = new Double[m1.length][m1[0].length];

        for (int j = 0; j < m2[0].length; j++) {
            for (int k = 0; k < m2.length; k++) {
                result[k][j] = m1[k][j] * m2[k][j];
            }
        }

        return result;
    }

    //Multiplicação de matrizes
    public static Double[][] Multiply(Double[][] m1, Double[][] m2, Double[][] bias) {
        Double[][] result = new Double[m1.length][m2[0].length];

        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m2[0].length; j++) {//j nao sai do 0
                Double sm = 0.0;
                for (int k = 0; k < m2.length; k++) {
                    sm += (m1[i][k] * m2[k][j]);
                }
                if (bias != null) {
                    result[i][j] = sm + bias[j][i];
                } else {
                    result[i][j] = sm;
                }

            }
        }

        return result;
    }

    //Método para printar matrizes.
    public static void PrintMatrix(Double[][] result) {
        DecimalFormat df = new DecimalFormat("0.0000000000");
        for (int i = 0; i < result.length; i++) {
            System.out.printf("[ ");
            for (int j = 0; j < result[0].length; j++) {
                System.out.printf(df.format(result[i][j]) + " ");
            }
            System.out.println("]");
        }

        System.out.println("");
    }

    //printa custos com módulo
    public static void PrintMatrixModulo(Double[][] result, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            System.out.printf("[ ");
            for (int j = 0; j < cols; j++) {
                if (result[i][j] < 0) {
                    System.out.printf(result[i][j] * -1 + " ");
                } else {
                    System.out.printf(result[i][j] + " ");
                }

            }
            System.out.println("]");
        }

        System.out.println("");
    }

    //multiplicação por escalar
    public static Double[][] MultiplyEscalar(Double[][] m1, Double escalar) {
        Double[][] result = new Double[m1.length][1];

        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m1[0].length; j++) {
                result[i][j] = (m1[i][j] * escalar);
            }
        }

        return result;
    }
    
    //inicializa a matriz de target, pegando valores vindos do csv
    public static Double[][] incializaTarget(Double[][] target, int row, int col) {

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                target[i][j] = 0.0;
            }
        }
        return target;
    }
    
    //faz o calculo do erro total da época
    public static Double somaTotal(Double[][] matrix) {
        Double result = 0.0;
        for (Double[] matrix1 : matrix) {
            if (matrix1[0] < 0) {
                result += (matrix1[0] * -1);
            } else {
                result += matrix1[0];
            }

        }
        return result;
    }

}
