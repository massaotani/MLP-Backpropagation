package mlp_backpropagation;

import java.util.Scanner;

public class Main {

    // Quando false (padrão), evita imprimir matrizes completas a cada amostra de treino -
    // isso é o que realmente tornava o programa lento/pesado, não o cálculo em si.
    // Mude para true se quiser inspecionar os valores intermediários durante o debug.
    public static boolean LOG_DETALHADO = false;

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

        System.out.println("Valor da taxa de aprendizado inicial (Valores entre 0 e 1. Pode usar virgula ou ponto.):");
        Double taxaAprendizado = Double.parseDouble(treino.next().replace(",", "."));

        System.out.println("Buscar valor inferior a qual erro médio no treino: (Pode usar virgula ou ponto.): ");
        Double erro = Double.parseDouble(treino.next().replace(",", "."));
        treino.nextLine(); // consome o restante da linha (quebra de linha pendente) antes do próximo nextLine()

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
        final int MAX_EPOCAS = 10000; //Limite de segurança: sem isso, um erro alvo (erro) muito baixo para os dados/arquitetura
        //escolhidos faz o treino rodar para sempre, pois o gradiente descendente "puro" (sem decaimento de
        //taxa de aprendizado) tende a oscilar perto de um mínimo em vez de convergir exatamente até qualquer valor
        final double LR_DECAY = 0.9995; //decaimento exponencial da taxa de aprendizado a cada epoca: comeca suave
        //(quase sem efeito nas primeiras centenas de epocas, entao ainda aprende rapido no inicio) e vai reduzindo
        //o "tamanho do passo" ao longo do treino, o que estabiliza a rede perto de um minimo em vez de ela ficar
        //oscilando/saindo de um bom ponto por causa de uma taxa de aprendizado fixa grande demais

        //Com pesos iniciais aleatorios, existe uma chance real (observada em testes: perto de
        //metade das vezes) da rede "travar" numa inicializacao ruim, onde alguns neuronios ficam
        //praticamente insensiveis a uma ou mais entradas e a rede nao consegue sair dali mesmo apos
        //milhares de epocas - o erro fica estacionado sem melhorar. Isso e um problema conhecido de
        //redes pequenas treinadas com gradiente descendente simples, nao um bug matematico: quando a
        //inicializacao e "boa", a mesma rede converge rapido e corretamente. A solucao padrao e
        //detectar a estagnacao e recomecar com pesos novos - como cerca de metade das inicializacoes
        //funcionam bem, a chance de precisar de muitos recomecos e baixa.
        final int JANELA_ESTAGNACAO = 300; //epocas sem melhora real antes de reiniciar
        final double MELHORA_MINIMA = 0.001; //melhora menor que isso nao conta como progresso
        final int MAX_REINICIOS = 20;
        Double melhorErro = Double.MAX_VALUE;
        int epocasSemMelhora = 0;
        int reinicios = 0;

        /**
         * **********INICIANDO TREINAMENTO************
         */
        while (erroMedio > erro && epoca < MAX_EPOCAS) {
            System.out.println("\n ----------------->> EPOCA: " + epoca + " <<-----------------\n");

            for (int i = 0; i < csv.calcLinhas(); i++) {

                if (LOG_DETALHADO) {
                    System.out.println("PESOS CAMADA ENTRADA -> OCULTA");
                    Matrix.PrintMatrix(rede.getPesosEntrada());

                    System.out.println("BIAS ENTRADA");
                    Matrix.PrintMatrix(rede.getBiasEntrada());

                    System.out.println("PESOS CAMADA OCULTA -> SAÍDA");
                    Matrix.PrintMatrix(rede.getPesosOculta());

                    System.out.println("BIAS CAMADA OCULTA");
                    Matrix.PrintMatrix(rede.getBiasOculta());
                }

                Double[][] aux = Matrix.povoaNeuronios(entrada, i);

                if (LOG_DETALHADO) {
                    System.out.println("NEURONIO ENTRADA:");
                    Matrix.PrintMatrix(aux);
                }
                aux = Matrix.transposta(aux); // Transforma em coluna

                Double[][] auxTarget = Matrix.povoaNeuronios(target, i);
                auxTarget = Matrix.transposta(auxTarget);

                /*Feed Foward  - Passo 1: Multiplica Neuronios da camada de entrada pelos pesos e faz um somatório. 
                Adiciona o Bias referente a cada Neuronio da Camada Oculta e aplica a função de ativação sigmoide*/
                Double[][] middleLayer = Operations.FeedForward(aux, rede.getPesosEntrada(), rede.getBiasEntrada());

                /*Feed Foward  - Passo 2: Multiplica Neuronios da camada oculta pelos pesos e faz um somatório. 
                Adiciona o Bias referente a cada Neuronio da Camada de Saída e aplica a função de ativação sigmoide*/
                Double[][] saida = Operations.FeedForward(middleLayer, rede.getPesosOculta(), rede.getBiasOculta());

                if (LOG_DETALHADO) {
                    System.out.println("\n ---------->> FEED FOWARD <<----------\n");
                    System.out.println("CAMADA DE SAÍDA - NEURONIOS ATIVADOS");
                    Matrix.PrintMatrix(saida);

                    System.out.println("TARGET:");
                    Matrix.PrintMatrix(auxTarget);
                    System.out.println("\n ---------->> BACK PROPAGATION <<----------\n");
                }
                erroTotal += Operations.backPropagation(saida, middleLayer, aux, auxTarget, rede);

            }

            //Calcula o erro medio de cada época dividindo o erro total pelo número de entradas
            erroMedio = erroTotal / nLinhas;
            System.out.println("FIM DA ÉPOCA " + epoca + ".\n CUSTO MÉDIO: " + erroMedio + "\n\n\n");

            epoca++;
            erroTotal = 0.0;
            rede.setTaxaAprendizado(rede.getTaxaAprendizado() * LR_DECAY);

            if (melhorErro - erroMedio > MELHORA_MINIMA) {
                melhorErro = erroMedio;
                epocasSemMelhora = 0;
            } else {
                epocasSemMelhora++;
            }

            if (epocasSemMelhora >= JANELA_ESTAGNACAO && erroMedio > erro && reinicios < MAX_REINICIOS) {
                reinicios++;
                System.out.println(">>> Treino estagnado (sem melhora ha " + JANELA_ESTAGNACAO
                        + " epocas). Reiniciando com nova inicializacao aleatoria (reinicio " + reinicios
                        + "/" + MAX_REINICIOS + ") <<<");
                rede = new RedeNeural(entrada[0].length, nSaidas, tamCamadaOculta, taxaAprendizado);
                erroMedio = 1.0;
                melhorErro = Double.MAX_VALUE;
                epocasSemMelhora = 0;
            }

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
        if (erroMedio <= erro) {
            System.out.println("TERMINO DO TREINO (convergiu em " + epoca + " epocas)");
        } else {
            System.out.println("TERMINO DO TREINO (limite de " + MAX_EPOCAS + " epocas atingido sem atingir o erro alvo; erro medio final: " + erroMedio + ")");
        }

        /**
         * **********INICIANDO TESTES************
         */
        // Reutiliza o mesmo Scanner (treino) em vez de criar um novo sobre System.in:
        // abrir um segundo Scanner sobre o mesmo fluxo faz com que os dados já
        // bufferizados pelo primeiro sejam perdidos, causando NoSuchElementException.
        System.out.println("PARA INICIAR OS TESTES FORNECA CAMINHO DO DATASET DE TESTES");
        String arquivoTestes = treino.nextLine();
        LeitorCSV csvTeste = new LeitorCSV(arquivoTestes);
        int nLinhasTeste = csvTeste.calcLinhas(); // usa o tamanho do dataset de TESTE, não o de treino

        Double[][] entradaTeste = new Double[nLinhasTeste][nAtributos];
        csvTeste.runEntrada(entradaTeste, nAtributos, nSaidas);

        for (int i = 0; i < entradaTeste.length; i++) {

            Double[][] aux = Matrix.povoaNeuronios(entradaTeste, i); // usa os dados de TESTE, não os de treino
            aux = Matrix.transposta(aux);

            Double[][] middleLayerTeste = Operations.FeedForward(aux, rede.getPesosEntrada(), rede.getBiasEntrada());
            Double[][] saidaTeste = Operations.FeedForward(middleLayerTeste, rede.getPesosOculta(), rede.getBiasOculta());

            System.out.println("Saida do Teste: " + i);
            Matrix.PrintMatrix(saidaTeste);
        }
        //colocar saida no csv

    }
}
