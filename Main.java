package mlp_backpropagation;

import java.util.Scanner;

public class Main {

    public static void main(String[] args) {

        System.out.println("REDE MLP");

        Scanner treino = new Scanner(System.in); //Scanner para valores de entrada

        System.out.println("Caminho do dataset de treino: ");
        String arquivoEntrada = treino.nextLine();

        System.out.println("Numero de neuronios na camada escondida: ");
        int tamCamadaOculta = treino.nextInt();

        System.out.println("Numero de atributos de camada entrada: ");
        int nAtributos = treino.nextInt(); //Recebe o número de valores por linha do dataset,
        //no caso, número de neurônios de entrada.

        System.out.println("Numero de saidas possiveis: "); //numero de neuronios de saida (target)
        int nSaidas = treino.nextInt();

        System.out.println("Valor da taxa de aprendizado inicial (Valores entre 0 e 1. Utilizar virgula e não ponto.):");
        Double taxaAprendizado = treino.nextDouble();

        System.out.println("Buscar valor inferior a qual erro médio no treino: (Utilizar virgula e não ponto): ");
        Double erro = treino.nextDouble();

        /*
        Ler toda a planilha csv e colocar as linhas em uma matriz de entradas.
        A partir dessa matriz de entrada, criar o laço que vai percorrer as epocas.
         */
        LeitorCSV csv = new LeitorCSV(arquivoEntrada);
        int nLinhas = csv.calcLinhas(); //Recebe número de linhas DE TODO O DATA SET, no caso, número de entradas.

        Double[][] entrada = new Double[nLinhas][nAtributos];
        csv.runEntrada(entrada, nAtributos, nSaidas);

        Double[][] target = new Double[entrada.length][1];
        target = csv.target;

        //Instanciando a rede neural com os parametros fornecidos
        RedeNeural rede = new RedeNeural(entrada[0].length, nSaidas, tamCamadaOculta, taxaAprendizado);

        Double erroTotal = 0.0; //Armazena a soma de erros DE CADA ÉPOCA
        Double erroMedio = 1.0; //Armazena a divisão do erro total de cada epoca pelo numero de iteracoes
        //fornecendo o erro medio de cada epoca 

        int epoca = 0;        //Armazena o numero de epocas decorridas        

        /**
         * **********INICIANDO TREINAMENTO************
         */
        while (erroMedio > erro) {
            System.out.println("\n ----------------->> EPOCA: " + epoca + " <<-----------------\n");

            for (int i = 0; i < csv.calcLinhas(); i++) {
               
                System.out.println("PESOS CAMADA ENTRADA -> OCULTA");
                Matrix.PrintMatrix(rede.getPesosEntrada());

                System.out.println("BIAS ENTRADA");
                Matrix.PrintMatrix(rede.getBiasEntrada());

                System.out.println("PESOS CAMADA OCULTA -> SAÍDA");
                Matrix.PrintMatrix(rede.getPesosOculta());

                System.out.println("BIAS CAMADA OCULTA");
                Matrix.PrintMatrix(rede.getBiasOculta());
                
                Double[][] aux = Matrix.povoaNeuronios(entrada, i);

                System.out.println("NEURONIO ENTRADA:");
                Matrix.PrintMatrix(aux);
                aux = Matrix.transposta(aux); // Transforma em coluna

                Double[][] auxTarget = Matrix.povoaNeuronios(target, i);
                auxTarget = Matrix.transposta(auxTarget);

                /*Feed Foward  - Passo 1: Multiplica Neuronios da camada de entrada pelos pesos e faz um somatório. 
                Adiciona o Bias referente a cada Neuronio da Camada Oculta e aplica a função de ativação sigmoide*/
                System.out.println("\n ---------->> FEED FOWARD <<----------\n");
                Double[][] middleLayer = Operations.FeedForward(aux, rede.getPesosEntrada(), rede.getBiasEntrada());
                System.out.println("PESOS CAMADA OCULTA -> SAÍDA");
                Matrix.PrintMatrix(rede.getPesosOculta());

                System.out.println("BIAS CAMADA OCULTA");
                Matrix.PrintMatrix(rede.getBiasOculta());

                /*Feed Foward  - Passo 2: Multiplica Neuronios da camada oculta pelos pesos e faz um somatório. 
                Adiciona o Bias referente a cada Neuronio da Camada de Saída e aplica a função de ativação sigmoide*/
                System.out.println("CAMADA DE SAÍDA - NEURONIOS ATIVADOS");
                Double[][] saida = Operations.FeedForward(middleLayer, rede.getPesosOculta(), rede.getBiasOculta());
                Matrix.PrintMatrix(saida);
                
                System.out.println("TARGET:");
                Matrix.PrintMatrix(auxTarget);
                System.out.println("\n ---------->> BACK PROPAGATION <<----------\n");
                erroTotal += Operations.backPropagation(saida, middleLayer, aux, auxTarget, rede);

            }

            //Calcula o erro medio de cada época dividindo o erro total pelo número de entradas
            erroMedio = erroTotal / nLinhas;
            System.out.println("FIM DA ÉPOCA " + epoca + ".\n CUSTO MÉDIO: " + erroMedio + "\n\n\n");

            epoca++;
            erroTotal = 0.0;
            if (erroMedio < erro) {
                System.out.println("Pesos Entrada Final");
                Matrix.PrintMatrix(rede.pesosEntrada);

                System.out.println("Pesos Camada Oculta Final");
                Matrix.PrintMatrix(rede.pesosOculta);

                System.out.println("Bias Entrada Final");
                Matrix.PrintMatrix(rede.biasEntrada);

                System.out.println("Bias Oculta Final");
                Matrix.PrintMatrix(rede.biasOculta);

            }
        }
        System.out.println("TERMINO DO TREINO");

        /**
         * **********INICIANDO TESTES************
         */
        Scanner teste = new Scanner(System.in);

        System.out.println("PARA INICIAR OS TESTES FORNECA CAMINHO DO DATASET DE TESTES");
        String arquivoTestes = teste.nextLine();
        LeitorCSV csvTeste = new LeitorCSV(arquivoTestes);

        Double[][] entradaTeste = new Double[nLinhas][nAtributos];
        csvTeste.runEntrada(entradaTeste, nAtributos, nSaidas);

        for (int i = 0; i < entradaTeste.length; i++) {

            Double[][] aux = Matrix.povoaNeuronios(entrada, i);
            aux = Matrix.transposta(aux);

            Double[][] middleLayerTeste = Operations.FeedForward(aux, rede.getPesosEntrada(), rede.getBiasEntrada());
            Double[][] saidaTeste = Operations.FeedForward(middleLayerTeste, rede.getPesosOculta(), rede.getBiasOculta());

            System.out.println("Saida do Teste: " + i);
            Matrix.PrintMatrix(saidaTeste);
        }
        //colocar saida no csv

    }
}
