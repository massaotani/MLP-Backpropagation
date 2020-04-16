package mlp_backpropagation;


// ESTRUTURA DA REDE NEURAL 
public class RedeNeural {
    
    //Hiper Parâmetros
    //Escolhidos pelo usuario na inicializaçao
    Double taxaAprendizado; //Valor inicial da taxa de aprendizado (valores entre 0 e 1)
    int tamCamadaOculta;           //Neuronios na camada escondida       
    //Setados de acordo com o arquivo de entrada fornecido
    int tamSaida;      //Neuronios de saida
    int tamCamadaEntrada;        //Numero de neuronios de entrada   
    
    //Estruturas que receberao os valores dos pesos... 
    Double[][] biasEntrada;     //Bias da cada intermediaria
    Double[][] biasOculta;     //Bias da cada de saida
    Double[][] pesosEntrada; //Pesos da camada intermediaria
    Double[][] pesosOculta;//Pesos da camada de saida   
    
    //...e o metodo que as inicializa de acordo com os hiper parametros
    public void initialize() {
        biasEntrada = new Double[1][this.tamSaida];
        biasOculta = new Double[1][this.tamSaida];
        pesosEntrada = new Double[this.tamCamadaOculta][this.tamCamadaEntrada];
        pesosOculta = new Double[this.tamCamadaOculta][this.tamSaida];
    }
    
    
    //CONSTRUTOR
    public RedeNeural(int tamCamadaEntrada, int tamSaida, int tamCamadaOculta, Double taxaAprendizado) {
        this.tamCamadaEntrada = tamCamadaEntrada;
        this.tamCamadaOculta = tamCamadaOculta;
        this.tamSaida = tamSaida;
        this.taxaAprendizado = taxaAprendizado;

        initialize();

        // Atribuição dos bias randomicos da primeira camada escondida
        this.biasEntrada = Matrix.Randomize(this.biasEntrada, 1, this.tamCamadaOculta);
        
        // Atribuicao dos bias randomicos da segunda camada escondida
        this.biasOculta = Matrix.Randomize(this.biasOculta, 1, this.tamSaida);

        // Atribuicao de pesos randomicos dos neuronios de cada entrada para cada um da camada
        this.pesosEntrada = Matrix.Randomize(this.pesosEntrada, this.tamCamadaOculta, this.tamCamadaEntrada);
               
        // Atribuicao de pesos dos randomicos neuronios da camada escondida 1 para os da camada
        this.pesosOculta = Matrix.Randomize(this.pesosOculta, this.tamCamadaOculta, this.tamSaida);
      
    }
    public Integer getTamCamadaEntrada() {
        return tamCamadaEntrada;
    }

    public void setTamCamadaEntrada(Integer tamCamadaEntrada) {
        this.tamCamadaEntrada = tamCamadaEntrada;
    }

    public Integer getTamSaida() {
        return tamSaida;
    }

    public void setTamSaida(Integer tamSaida) {
        this.tamSaida = tamSaida;
    }

    public Integer getTamCamadaOculta() {
        return tamCamadaOculta;
    }

    public void setTamCamadaOculta(Integer tamCamadaOculta) {
        this.tamCamadaOculta = tamCamadaOculta;
    }

    public Double getTaxaAprendizado() {
        return taxaAprendizado;
    }

    public Double[][] getBiasEntrada() {
        return biasEntrada;
    }

    public void setBiasEntrada(Double[][] biasEntrada) {
        this.biasEntrada = biasEntrada;
    }

    public Double[][] getBiasOculta() {
        return biasOculta;
    }

    public void setBiasOculta(Double[][] biasOculta) {
        this.biasOculta = biasOculta;
    }

    public Double[][] getPesosEntrada() {
        return pesosEntrada;
    }

    public void setPesosEntrada(Double[][] pesosEntrada) {
        this.pesosEntrada = pesosEntrada;
    }

    public Double[][] getPesosOculta() {
        return pesosOculta;
    }

    public void setPesosOculta(Double[][] pesosOculta) {
        this.pesosOculta = pesosOculta;
    }

}
