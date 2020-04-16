package mlp_backpropagation;

public class Operations {

    public static Double[][] FeedForward(Double[][] entradas, Double[][] pesos, Double[][] bias) {

        /*faz uma multiplicação de matrizes pesos x entradas e soma o bias ao final da iteração de cada neuronio*/
        Double[][] proximaLayer = Matrix.Multiply(pesos, entradas, bias);

        //Aplica função de ativação
        proximaLayer = FuncaoDeAtivacao(proximaLayer);
        

        return proximaLayer;

    }

    /*Nos comentários abaixo foram usadas as notações apresentadas nos slides do backPropagation, para que a explicação fique mais fácil*/
    public static Double backPropagation(Double[][] saida, Double[][] oculta, Double[][] entrada, Double[][] esperado, RedeNeural rede) {

        //Calcular a taxa de erro, subtraindo a saida do esperado, para saber "o quao distante" esta da resposta esperada
        //(Tk - Yk)
        Double[][] matrizDeErros = Matrix.subtract(esperado, saida); //matriz que armazena a informaçao de erro de cada neuronio de saida

        System.out.println("CUSTOS");
        Matrix.PrintMatrixModulo(matrizDeErros, matrizDeErros.length, matrizDeErros[0].length);

        Double somaErros = Matrix.somaTotal(matrizDeErros);                 //Somatorio dos erros da saida, ou seja, a saida
        System.out.println("ERRO TOTAL DA SAIDA: " + somaErros + "\n");         //da funcao de custos

        //Armazena os valores resultantes de cada saída na funcao Sigmoide derivada (F'(Y_INk))
        Double[][] sigmoideDerivada = FuncaoDeAtivacaoDerivada(saida);

        //Calculando termos de informacao de erro
        /*Termo de informação do erro:
        Foi utilizado o Produto Hadamard para obtenção do gradiente de erro, uma vez que o intuito não
        é fazer uma multiplicação de matrizes tradicional e sim multiplicar os valores de taxaErro[i][j](tamanho 7x1)
        pelos respectivos valores sigmoideDerivada[i][j](tamanho 7x1), para gerar uma nova matriz com o mesmo tamanho(7x1),
        para ter o termo de informação de erro correto para cada neuronio da camada oculta
         */
        //Calculo do gradiente de erro (Delta K = (Tk - Yk)* F'(Y_INk))
        Double[][] infoErroSaida = Matrix.ProdutoHadamard(sigmoideDerivada, matrizDeErros);

        //Multiplicar Delta K pela taxa de Aprendizado (Multiplicacao de cada termo pelo escalar (taxaErro))
        //Feito para dimensionar o 'tamanho do passo' de correcao
        infoErroSaida = Matrix.MultiplyEscalar(infoErroSaida, rede.getTaxaAprendizado());

        //Novo peso do Bias Middle Layer = pesos antigos + delta dos erros 
        //Matricialmente adciona-se aos pesos antigos a matriz transposta do gradiente de erro
        rede.setBiasOculta((Matrix.Add(Matrix.transposta(infoErroSaida), rede.getBiasOculta())));

        //W = (Delta K * taxaAprendizado)*neuronios da oculta       
        //  = (Matriz gradienteErro * matriz transposta da oculta)
        Double[][] ocultaTransposta = Matrix.transposta(oculta); //Transposta da oculta para fazer a multiplicacao
        Double[][] w = Matrix.Multiply(infoErroSaida, ocultaTransposta, null); //null: multiplicacao comum entre matrizes
        //isto é, nao esta sendo usado para o calculo do bias
        //como fazemos no comeco do feedfoward

        //Novos pesos da camada oculta para saida = pesos antigos + deltas ja calculados (w)     
        rede.setPesosOculta(Matrix.Add(w, rede.getPesosOculta()));

        /////////CALCULO E CORRECAO DE PESOS: OCULTA <-> ENTRADA/////////////
        // TRANSPOSTA DA OCULTA 
        // Double[][] pesosHiddenTransposta = Matrix.transposta(w);
        Double[][] pesosHiddenTransposta = Matrix.transposta(rede.getPesosOculta());

        //Multiplica W pela taxa de Erro para calcular a taxa de erro da camada oculta
        //Delta INj = Matriz W * Matriz Taxa Erro
        Double[][] errosOculta = Matrix.Multiply(pesosHiddenTransposta, matrizDeErros, null);

        //CALCULO DO GRADIENTE DE ACORDO COM O ERRO DA OCULTA
        Double[][] sigmoideDerivada2 = FuncaoDeAtivacaoDerivada(oculta); //DERIVADA SIGMOIDE DA OCULTA
        
        /*transpoe a entrada para transforma-la numa matriz linha, para posteriormente ser usada no
        metodo de multiplicação*/
        Double[][] entradaTransposta = Matrix.transposta(entrada);

        //Delta J = Delta IN j * F'(Z_INj), ou seja 
        Double[][] infoErroOculta = Matrix.ProdutoHadamard(sigmoideDerivada2, errosOculta);

        // Delta J * taxaAprendizado
        infoErroOculta = Matrix.MultiplyEscalar(infoErroOculta, rede.getTaxaAprendizado());
        System.out.println("TERMO DE INFORMAÇÃO DE  ERRO CAMADA OCULTA * TX DE APRENDIZADO");
        Matrix.PrintMatrix(infoErroOculta);

        //Novo peso do Bias Layer Entrada = peso antigo + delta do erro
        rede.setBiasEntrada(Matrix.Add(Matrix.transposta(infoErroOculta), rede.getBiasEntrada()));

        //Novo peso da Middle Layer = peso antigo + delta do erro
        Double[][] j = Matrix.Multiply(infoErroOculta, entradaTransposta, null);
        
        //Novo peso da Camada de Entrada = peso antigo + delta do erro
        rede.setPesosEntrada(Matrix.Add(j, rede.getPesosEntrada())); //N TEM QUE SER SUBTRAÃ‡ÃƒO????

        return somaErros;
    }
    
    /*Função de ativação sigmoide binária, apresentada nos slides em aula*/
    public static Double[][] FuncaoDeAtivacao(Double[][] entrada) {
        Double[][] aux = new Double[entrada.length][entrada[0].length];

        for (int i = 0; i < entrada.length; i++) {
            aux[i][0] = (Double) (1.0 / (1.0 + Math.pow(Math.E, (-1.0 * entrada[i][0]))));
        }

        return aux;
    }
    //Função sigmoide derivada, para uso no backpropagation
    public static Double[][] FuncaoDeAtivacaoDerivada(Double[][] entrada) {
        Double[][] aux = new Double[entrada.length][entrada[0].length];

        for (int i = 0; i < entrada.length; i++) {
            Double e = (Double) (1.0 / (1.0 + Math.pow(Math.E, (-1.0 * entrada[i][0]))));
            aux[i][0] = (Double) (e * (1.0 - e));
        }

        return aux;
    }
}
