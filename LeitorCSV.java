package mlp_backpropagation;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import java.io.IOException;

//Operações sobre os arquivos
public class LeitorCSV {

    String arquivoCSV;
    public int linhas;
    public int colunas;
    public Double[][] target;

    public LeitorCSV(String arquivoCSV) {
        this.arquivoCSV = arquivoCSV;
    }

    public void MatrizToString(Double[][] entrada) {
        Double entrada2[] = new Double[entrada[0].length];
        for (int i = 0; i < entrada.length; i++) {
            for (int j = 0; j < entrada[0].length; j++) {
                entrada2[j] = entrada[i][j];
            }
        }
    }

    public int calcLinhas() {

        BufferedReader br = null;
        String linha = "";
        int contadorLinha = 0;
        try {

            br = new BufferedReader(new FileReader(arquivoCSV));
            while ((linha = br.readLine()) != null) {
                contadorLinha++;
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return contadorLinha;
    }

    public void runEntrada(Double[][] entrada, int row, int col) {

        BufferedReader br = null;
        String linha = "";
        String csvDivisor = ",";
        target = new Double[row][col];

        Matrix.incializaTarget(target, row, col);
        try {

            int contadorLinha = 0;
            br = new BufferedReader(new FileReader(arquivoCSV));
            while ((linha = br.readLine()) != null) {

                String[] letra = linha.split(csvDivisor);
                for (int i = 0; i < letra.length; i++) {
                    if (letra[i].contains("-")) {
                        if (letra[i].contains(".")) {
                            String letraReplace = letra[i].replace("-", "");
                            entrada[contadorLinha][i] = -1.0 * Double.parseDouble(letraReplace);

                        } else {
                            entrada[contadorLinha][i] = -1.0;
                        }

                    } else if (letra[i].equals("1") || letra[i].equals("0") || letra[i].equals("1.5")) {
                        entrada[contadorLinha][i] = Double.valueOf(letra[i]);

                    } else {
                        target = povoaTarget(letra[i], contadorLinha, target);
                    }
                }
                contadorLinha++;
            }

            MatrizToString(entrada);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public Double[][] povoaTarget(String letra, int indice, Double[][] target) {
        switch (letra) {
            case "A":
                target[indice][0] = Double.parseDouble("1");
                break;
            case "B":
                target[indice][1] = Double.parseDouble("1");
                break;
            case "C":
                target[indice][2] = Double.parseDouble("1");
                break;
            case "D":
                target[indice][3] = Double.parseDouble("1");
                break;
            case "E":
                target[indice][4] = Double.parseDouble("1");
                break;
            case "J":
                target[indice][5] = Double.parseDouble("1");
                break;
            case "K":
                target[indice][6] = Double.parseDouble("1");
                break;
        }

        return target;
    }

}
