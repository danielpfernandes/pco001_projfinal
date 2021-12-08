/*******************************************************
 * Aplicação C++ para a classificação usando algoritmo *
 * OPF, parte do projeto final de PCO001 da            *
 * Universidade Federal de Itajubá - MG                *
 *                                                     *
 *                                                     *
 * Projeto original: Thierry Moreira, 2019             *
 *                                                     *
 * Adaptado por: Daniel P Fernandes, Natalia S         *
 * Sanchez & Alexandre L Sousa                         *
 *                                                     *
 *******************************************************/

// Copyright 2021 Daniel P Fernandes, Natalia S Sanchez & Alexandre
// L Sousa
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPF_HPP
#define OPF_HPP

#include <functional>
#include <stdexcept>
#include <algorithm>
#include <typeinfo>
#include <sstream>
#include <utility>
#include <cstring>
#include <string>
#include <limits>
#include <memory>
#include <vector>
#include <cmath>
#include <map>
#include <set>

#include <omp.h>

namespace opf
{

using uchar = unsigned char;
#define INF std::numeric_limits<float>::infinity()
#define NIL -1

// Função genérica de distância
template <class T>
using funcaoDistancia = std::function<T (const T*, const T*, size_t)>;



/*****************************************/
/*************** Binary IO ***************/
/*****************************************/

////////////
// Tipos de OPF
enum Tipo : unsigned char
{
    Classificador = 1,
    Agrupamento = 2,
};

//////////////////////
// Flags de Serialização
enum SFlags : unsigned char
{
    Supervisionado_SalvaPrototipos = 1,
    NaoSupervisionado_Anomalia = 2,
};

///////////////
// Funções de IO
/**
 * Escreve em saída em formato binário
 * @tparam T
 * @param saida Saída (output)
 * @param val valor a ser escrito
 */
template <class T>
void escreveBinario(std::ostream& saida, const T& val)
{
    saida.write((char*) &val, sizeof(T));
}

/**
 * Escreve em saída em formato binário
 * @tparam T
 * @param saida Saída (output)
 * @param val valor a ser escrito
 * @param n número de referência ao valor a ser escrito (tamanho)
 */
template <class T>
void escreveBinario(std::ostream& saida, const T* val, int n= 1)
{
    saida.write((char*) val, sizeof(T) * n);
}

/**
 * Faz a leitura de dados binários
 * @tparam T
 * @param entrada Entrada (input)
 * @return Dado convertido no tipo de origem
 */
template <class T>
T lerBinario(std::istream& entrada)
{
    T val;
    entrada.read((char*) &val, sizeof(T));
    return val;
}

/**
 * Faz a leitura de dados binários
 * @tparam T
 * @param entrada Entrada (entrada)
 * @param val valor do dado
 * @param n referência (tamanho)
 */
template <class T>
void lerbinario(std::istream& entrada, T* val, int n= 1)
{
    entrada.read((char*) val, sizeof(T) * n);
}

/*****************************************/
/************** Matrix type **************/
/*****************************************/
/**
 * Classe n-dimensional de matriz densa
 * @tparam T
 */
template <class T=float>
class Mat
{
protected:
    std::shared_ptr<T> dado;
public:
    size_t linhas, colunas;
    size_t tamanho;
    size_t passo;

    // Construtores
    Mat();
    Mat(const Mat<T>& other);
    Mat(size_t linhas, size_t colunas);
    Mat(size_t linhas, size_t colunas, T val0);
    Mat(std::shared_ptr<T>& dado, size_t linhas, size_t colunas, size_t passo = 0);
    Mat(T* dado, size_t linhas, size_t colunas, size_t passo=0);

    // Protótipos
    virtual T& em(size_t i, size_t j);
    const virtual T em(size_t i, size_t j) const;
    virtual T* linha(size_t i);
    const virtual T* linha(size_t i) const;
    virtual T* operator[](size_t i);
    const virtual T* operator[](size_t i) const;
    Mat<T>& operator=(const Mat<T>& outro);
    virtual Mat<T> copia();

    void libera();
};

template <class T>
Mat<T>::Mat()
{
    this->linhas = this->colunas = this->tamanho = this->passo = 0;
}

template <class T>
Mat<T>::Mat(const Mat<T>& other)
{
    this->linhas = other.linhas;
    this->colunas = other.colunas;
    this->tamanho = other.tamanho;
    this->dado = other.dado;
    this->passo = other.passo;
}

template <class T>
Mat<T>::Mat(size_t linhas, size_t colunas)
{
    this->linhas = linhas;
    this->colunas = colunas;
    this->tamanho = linhas * colunas;
    this->passo = colunas;
    this->dado = std::shared_ptr<T>(new T[this->tamanho], std::default_delete<T[]>());
}

template <class T>
Mat<T>::Mat(size_t linhas, size_t colunas, T val)
{
    this->linhas = linhas;
    this->colunas = colunas;
    this->tamanho = linhas * colunas;
    this->passo = colunas;
    this->dado = std::shared_ptr<T>(new T[this->tamanho], std::default_delete<T[]>());

    for (size_t i = 0; i < linhas; i++)
    {
        T* linha = this->linha(i);
        for (size_t j = 0; j < colunas; j++)
            linha[j] = val;
    }
}

template <class T>
Mat<T>::Mat(std::shared_ptr<T>& dado, size_t linhas, size_t colunas, size_t passo)
{
    this->linhas = linhas;
    this->colunas = colunas;
    this->tamanho = linhas * colunas;
    this->dado = dado;
    if (passo)
        this->passo = passo;
    else
        this->passo = colunas;
}

// Recebe um ponteiro para algum dado, que pode não ser excluído.
template <class T>
Mat<T>::Mat(T* dado, size_t linhas, size_t colunas, size_t passo)
{
    this->linhas = linhas;
    this->colunas = colunas;
    this->tamanho = linhas * colunas;
    this->dado = std::shared_ptr<T>(dado, [](T *p) {});
    if (passo)
        this->passo = passo;
    else
        this->passo = colunas;
}

template <class T>
T& Mat<T>::em(size_t i, size_t j)
{
    size_t indice = i * this->passo + j;
    return this->dado.get()[indice];
}

template <class T>
const T Mat<T>::em(size_t i, size_t j) const
{
    size_t indice = i * this->passo + j;
    return this->dado.get()[indice];
}

template <class T>
T* Mat<T>::linha(size_t i)
{
    size_t indice = i * this->passo;
    return this->dado.get() + indice;
}

template <class T>
const T* Mat<T>::linha(size_t i) const
{
    size_t indice = i * this->passo;
    return this->dado.get() + indice;
}

template <class T>
T* Mat<T>::operator[](size_t i)
{
    size_t indice = i * this->passo;
    return this->dado.get() + indice;
}

template <class T>
const T* Mat<T>::operator[](size_t i) const
{
    size_t indice = i * this->passo;
    return this->dado.get() + indice;
}

template <class T>
Mat<T>& Mat<T>::operator=(const Mat<T>& outro)
{
    if (this != &outro)
    {
        this->linhas = outro.linhas;
        this->colunas = outro.colunas;
        this->tamanho = outro.tamanho;
        this->dado = outro.dado;
        this->passo = outro.passo;
    }

    return *this;
}

template <class T>
Mat<T> Mat<T>::copia()
{
    Mat<T> out(this->linhas, this->colunas);
    for (size_t i = 0; i < this->linhas; i++)
    {
        T* linha = this->linha(i);
        T* linhaExterna = out.linha(i);
        for (size_t j = 0; j < this->colunas; j++)
            linhaExterna[j] = linha[j];
    }

    return std::move(out);
}

template <class T>
void Mat<T>::libera()
{
    this->dado.reset();
}

/*****************************************/


// Função de distância padrão
template <class T>
T distanciaEuclidiana(const T* a, const T* b, size_t tamanho)
{
    T soma = 0;
    for (size_t i = 0; i < tamanho; i++)
    {
        soma += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return (T)sqrt(soma);
}

template <class T>
T magnitude(const T* v, size_t tamanho)
{
    T soma = 0;
    for (size_t i = 0; i < tamanho; i++)
    {
        soma += v[i] * v[i];
    }
    return (T)sqrt(soma);
}

// Uma função de distância alternativa
template <class T>
T distanciaDeCosseno(const T* a, const T* b, size_t tamanho)
{
    T dividendo = 0;
    for (size_t i = 0; i < tamanho; i++)
    {
        dividendo += a[i] * b[i];
    }

    T divisor = magnitude<T>(a, tamanho) * magnitude<T>(b, tamanho);

    // 1 - similaridade cosseno
    return 1 - (dividendo / divisor);
}

template <class T>
Mat<T> computaDistanciasDeTreinamento(const Mat<T> &caracteristicas, funcaoDistancia<T> distancia= distanciaEuclidiana<T>)
{
    Mat<float> distancias(caracteristicas.linhas, caracteristicas.linhas);

    #pragma omp parallel for shared(caracteristicas, distancias)
    for (size_t i = 0; i < caracteristicas.linhas - 1; i++)
    {
        distancias[i][i] = 0;
        for (size_t j = i + 1; j < caracteristicas.linhas; j++)
        {
            distancias[i][j] = distancias[j][i] = distancia(caracteristicas[i], caracteristicas[j], caracteristicas.colunas);
        }
    }

    return distancias;
}


/************************************************/
/********* Tipo de matriz de distância **********/
/************************************************/
// Em vez de armazenar elementos n x n, só armazenamos o triângulo superior,
// que tem (n -1)/2 elementos (menos da metade).
template <class T>
class MatrizDeDistancias: public Mat<T>
{
private:
    T diag_vals = static_cast<T>(0);
    int buscaIndice(int i, int j) const;
public:
    MatrizDeDistancias(){ this->linhas = this->colunas = this->tamanho = 0;};
    MatrizDeDistancias(const MatrizDeDistancias& outro);
    MatrizDeDistancias(const Mat<T>& caracteristicas, funcaoDistancia<T> distancia=distanciaEuclidiana<T>);
    virtual T& em(size_t i, size_t j);
    const virtual T at(size_t i, size_t j) const;
};

// A primeira linha tem n-1 colunas, a segunda tem n-2, e assim por diante até linha n tem 0 colunas.
// Assim
#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))
template <class T>
inline int MatrizDeDistancias<T>::buscaIndice(int i, int j) const
{
    if (i > j)
        SWAP(i, j);
    return ((((this->linhas << 1) - i - 1) * i) >> 1) + (j - i - 1);
}

template <class T>
MatrizDeDistancias<T>::MatrizDeDistancias(const MatrizDeDistancias& outro)
{
    this->linhas = outro.linhas;
    this->colunas = outro.colunas;
    this->tamanho = outro.tamanho;
    this->dado = outro.dado;
}

template <class T>
MatrizDeDistancias<T>::MatrizDeDistancias(const Mat<T>& caracteristicas, funcaoDistancia<T> distancia)
{
    this->linhas = caracteristicas.linhas;
    this->colunas = caracteristicas.linhas;
    this->tamanho = (this->linhas * (this->linhas - 1)) / 2;
    this->dado = std::shared_ptr<T>(new float[this->tamanho], std::default_delete<float[]>());
    for (size_t i = 0; i < this->linhas; i++)
    {
        for (size_t j = i+1; j < this->linhas; j++)
            this->dado.get()[buscaIndice(i, j)] = distancia(caracteristicas[i], caracteristicas[j], caracteristicas.colunas);
    }
}

template <class T>
T& MatrizDeDistancias<T>::em(size_t i, size_t j)
{
    if (i == j)
        return this->diag_vals = static_cast<T>(0);
    return this->dado.get()[this->buscaIndice(i, j)];
}

template <class T>
const T MatrizDeDistancias<T>::at(size_t i, size_t j) const
{
    if (i == j)
        return 0;
    return this->dado.get()[this->buscaIndice(i, j)];
}


/*********************************************/
/************ Estruturas de Dados ************/
/*********************************************/

/**
 * Códigos de cores para o algoritmo de Prim
 */
enum Color{
    BRANCO, // Novo vértice
    CINZA,  // Na heap
    PRETO  // Já visto
};

/**
 * Classe simples para armazenar informações de vértice
 */
class Vertice
{
public:
    Vertice()
    {
        this->cor = BRANCO;
        this->predecessor = -1;
        this->custo = INF;
        this->isPrototipo = false;
    }

    size_t indice;      // Índice na lista - facilita as pesquisas
    Color cor;          // Cor na heap. branco: nunca visitada, cinza: na heap, preto: removido da heap
    float custo;        // Custo para alcançar o vértice
    int rotuloVerdade;  // Valor de referência
    int rotulo;         // Rótulo atribuído
    int predecessor;    // Vértice antecessor
    bool isPrototipo;   // Se o vértice é um protótipo
};

/**
 * Estrutura de dados de heap para usar como uma fila de prioridade
 *
 */
class heap
{
private:
    std::vector<Vertice> *vertices; // Uma referência para o vetor de contêiner original
    std::vector<Vertice*> vec;   // Um vetor de ponteiros para construir a heap em cima

    static bool comparaElemento(const Vertice* estruturaHeapEsquerda, const Vertice* estruturaHeapDireita)
    {
        return estruturaHeapEsquerda->custo >= estruturaHeapDireita->custo;
    }

public:
    // Construtor de tamanho
    heap(std::vector<Vertice> *vertices, const std::vector<int> &rotulos)
    {
        this->vertices = vertices;
        size_t n = vertices->size();
        this->vec.reserve(n);
        for (size_t i = 0; i < n; i++)
        {
            (*this->vertices)[i].indice = i;
            (*this->vertices)[i].rotuloVerdade = (*this->vertices)[i].rotulo = rotulos[i];
        }
    }
    /**
     * Insere o novo elemento na heap
     * @param item elemento
     * @param custo custo
     */
    void push(int item, float custo)
    {
        // Atualiza o valor de custo do vértice
        (*this->vertices)[item].custo = custo;

        // Já presente na heap
        if ((*this->vertices)[item].cor == CINZA)
            make_heap(this->vec.begin(), this->vec.end(), comparaElemento); // Remake the heap

        // Novo na heap
        else if ((*this->vertices)[item].cor == BRANCO)
        {
            (*this->vertices)[item].cor = CINZA;
            this->vec.push_back(&(*this->vertices)[item]);
            push_heap(this->vec.begin(), this->vec.end(), comparaElemento); // Push new item to the heap
        }
        // Note that black items can not be inserted into the heap
    }

    /**
     * Atualiza o custo do item sem atualizar a heap
     * @param item item
     * @param custo custo
     */
    void atualizaCusto(int item, float custo)
    {
        // Atualiza o valor de custo do vértice
        (*this->vertices)[item].custo = custo;
        if ((*this->vertices)[item].cor == BRANCO)
        {
            (*this->vertices)[item].cor = CINZA;
            this->vec.push_back(&(*this->vertices)[item]);
        }
    }

    /**
     * Atualiza a heap.
     * Isso é usado após várias chamadas para atualizaCusto, a fim de reduzir o número de chamadas para make_heap.
     */
    void heapify()
    {
        make_heap(this->vec.begin(), this->vec.end(), comparaElemento); // Remake the heap
    }

    /**
     * Remova e devolva o primeiro elemento da heap
     * @return primeiro elemento da heap
     */
    int pop()
    {
        // Obtém e marca o primeiro elemento
        Vertice *frente = this->vec.front();
        frente->cor = PRETO;
        // O remove da heap
        pop_heap(this->vec.begin(), this->vec.end(), comparaElemento);
        this->vec.pop_back();
        // E o retorna
        return frente->indice;
    }

    /**
     * Verifica se a heap está vazia
     * @return true se estive vazia
     */
    bool isVazia()
    {
        return this->vec.size() == 0;
    }

    /**
     * Verifica tamanho da heap
     * @return tamanho da heap
     */
    size_t size()
    {
        return this->vec.size();
    }
};

/*****************************************/



/*****************************************/
/****************** OPF ******************/
/*****************************************/

/******** Supervisionado ********/
template <class T=float>
class OPFSupervisionado
{
private:
    // Modelo
    Mat<T> dadoDeTreinamento; // Dados de treinamento (vetores originais ou matriz de distância)
    std::vector<Vertice> vertices;  // Modelo aprendido

    // Lista de vértices ordenados pelo custo. Útil para acelerar a classificação
    // Não é definido com size_t para reduzir o uso de memória, uma vez que o ML pode lidar com dados grandes
    std::vector<unsigned int> nodosOrdenados;

    // Opções
    bool isPrecomputado;
    funcaoDistancia<T> distancia;

    void prototipoDePrim(const std::vector<int> &rotulos);


public:
    OPFSupervisionado(bool isPrecomputado=false, funcaoDistancia<T> distancia=distanciaEuclidiana<T>);

    void ajusta(const Mat<T> &dadoDeTreinamento, const std::vector<int> &rotulos);
    std::vector<int> prediz(const Mat<T> &dadoDeTeste);

    bool getPrecomputado() {return this->isPrecomputado;}

    // Funções de serialização
    std::string serializa(uchar flags= 0);
    static OPFSupervisionado<T> desserializa(const std::string& conteudos);

    // Informações de treinamento
    std::vector<std::vector<float>> getPrototipos();
};

template <class T>
OPFSupervisionado<T>::OPFSupervisionado(bool isPrecomputado, funcaoDistancia<T> distancia)
{
    this->isPrecomputado = isPrecomputado;
    this->distancia = distancia;
}

/**
 * - O primeiro passo no procedimento de treinamento da OPF. Encontra o vértices protótipos
 *   usando o algoritmo da Árvore Geradora Mínima de Prim.
 * - Qualquer vértice com um vértice adjacente de uma classe diferente é tomado como um protótipo.
 * @tparam T
 * @param rotulos Rótulos dos dados
 */
template <class T>
void OPFSupervisionado<T>::prototipoDePrim(const std::vector<int> &rotulos)
{
    this->vertices = std::vector<Vertice>(this->dadoDeTreinamento.linhas);
    heap cabeca(&this->vertices, rotulos); // heap como uma fila de prioridades

    // Primeiro vértice arbitrário
    cabeca.push(0, 0);

    while(!cabeca.isVazia())
    {
        // Pega a cabeça da heap e marca-a preta
        size_t s = cabeca.pop();

        // Definição de protótipo
        int pred = this->vertices[s].predecessor;
        if (pred != NIL)
        {
            // Encontra pontos na fronteira entre duas classes...
            if (this->vertices[s].rotuloVerdade != this->vertices[pred].rotuloVerdade)
            {
                // E os define como protótipos
                this->vertices[s].isPrototipo = true;
                this->vertices[pred].isPrototipo = true;
            }
        }


        // Seleção de arestas
        #pragma omp parallel for default(shared)
        for (size_t t = 0; t < this->vertices.size(); t++)
        {
            // Se os vértices são diferentes e t não foi retirado da heap (marcado preto)
            if (s != t && this->vertices[t].cor != PRETO)
            {
                // Calcula o peso
                float peso;
                if (this->isPrecomputado)
                    peso = this->dadoDeTreinamento[s][t];
                else
                    peso = this->distancia(this->dadoDeTreinamento[s], this->dadoDeTreinamento[t], this->dadoDeTreinamento.colunas);

                // Atribui se menor do que o valor atual
                if (peso < this->vertices[t].custo)
                {
                    this->vertices[t].predecessor = static_cast<int>(s);
                    // cabeca.push(t, peso);
                    #pragma omp critical(updateHeap)
                    cabeca.atualizaCusto(t, peso);
                }
            }
        }
        cabeca.heapify();
    }
}

/**
 * Treina o modelo com o dado dado e rotulos.
 * @tparam T
 * @param dadoDeTreinamento
 *          - vetores de recurso original [n_samples, n_features] -- se isPrecomputado == false
 *          - matriz de distancia [n_samples, n_samples] -- se isPrecomputado == true
 * @param rotulos
 *          - valores dos rotulo-verdade [n_samples]
 */
template <class T>
void OPFSupervisionado<T>::ajusta(const Mat<T> &dadoDeTreinamento, const std::vector<int> &rotulos)
{
    if ((size_t)dadoDeTreinamento.linhas != rotulos.size())
        throw std::invalid_argument("[OPF/ajusta] Erro: tamanho dos dados não correspondem ao tamanho dos rótulos"
        + std::to_string(dadoDeTreinamento.linhas) + " x " + std::to_string(rotulos.size()));

    // Armazenar referência de dados para testes
    this->dadoDeTreinamento = dadoDeTreinamento;

    // Modelo de inicialização
    this->prototipoDePrim(rotulos); // Encontra protótipos
    heap cabeca(&this->vertices, rotulos); // heap como uma fila de prioridade

    // Inicialização
    for (Vertice& nodo: this->vertices)
    {
        nodo.cor = BRANCO;
        // Protótipos custo 0, não ter antecessor e povoar o heap
        if (nodo.isPrototipo)
        {
            nodo.predecessor = NIL;
            nodo.custo = 0;
        }
        else // Outros vértices iniciam com custo = INF
        {
            nodo.custo = INF;
        }
        // Uma vez que todos os vértices estão conectados a todos os outros
        cabeca.push(nodo.indice, nodo.custo);
    }

    // Lista de vértices ordenados por custo
    // Útil para acelerar a classificação
    this->nodosOrdenados.reserve(this->vertices.size());

    // consome a fila
    while(!cabeca.isVazia())
    {
        int s = cabeca.pop();
        this->nodosOrdenados.push_back(s);

        // Itera sobre todos os vizinhos
        #pragma omp parallel for default(shared)
        for (int t = 0; t < (int) this->vertices.size(); t++)
        {
            if (s != t && this->vertices[s].custo < this->vertices[t].custo)
            {
                // Computa o peso
                float peso;
                if (isPrecomputado)
                    peso = this->dadoDeTreinamento[s][t];
                else
                    peso = distancia(this->dadoDeTreinamento[s], this->dadoDeTreinamento[t], this->dadoDeTreinamento.colunas);

                float cost = std::max(peso, this->vertices[s].custo);
                if (cost < this->vertices[t].custo)
                {
                    this->vertices[t].predecessor = s;
                    this->vertices[t].rotulo = this->vertices[s].rotuloVerdade;

                    #pragma omp critical(updateHeap)
                    cabeca.atualizaCusto(t, cost);
                }
            }
        }
        cabeca.heapify();
    }
}

/**
 * Classifique um conjunto de amostras usando um modelo treinado por OPFSupervisionado::ajusta.
 * @tparam T
 * @param dadoDeTeste
 *          - vetores de recurso original [n_samples, n_features] -- se isPrecomputado == false
 *          - matriz de distancia [n_samples, n_samples] -- se isPrecomputado == true
 * @return
 * - previsões:
 * - vetor<int> de tamanho [n_test_samples] com saídas de classificação.
 */
template <class T>
std::vector<int> OPFSupervisionado<T>::prediz(const Mat<T> &dadoDeTeste)
{
    int n_amostrasDeTeste = (int) dadoDeTeste.linhas;
    int n_amostrasDeTreinamento = (int) this->vertices.size();

    // Previsões de saída
    std::vector<int> previsoes(n_amostrasDeTeste);

    #pragma omp parallel for default(shared)
    for (int i = 0; i < n_amostrasDeTeste; i++)
    {
        int indice = this->nodosOrdenados[0];
        int diceMinimo = 0;
        T custoMinimo = INF;
        T peso = 0;

        // 'nodosOrdenados' contém índices amostrais ordenados pelo custo, portanto,
        // se a melhor conexão atual custa menos do que o próximo vértice, é inútil continuar procurando.
        for (int j = 0; j < n_amostrasDeTreinamento && custoMinimo > this->vertices[indice].custo; j++)
        {
            // Obtém o próximo vértice na lista ordenada
            indice = this->nodosOrdenados[j];

            // Calcular sua distância até o ponto de consulta
            if (isPrecomputado)
                peso = dadoDeTeste[i][indice];
            else
                peso = distancia(dadoDeTeste[i], this->dadoDeTreinamento[indice], this->dadoDeTreinamento.colunas);

            // O custo corresponde ao máximo entre a distância e o custo de referência
            float custo = std::max(peso, this->vertices[indice].custo);

            if (custo < custoMinimo)
            {
                custoMinimo = custo;
                diceMinimo = indice;
            }
        }

        previsoes[i] = this->vertices[diceMinimo].rotulo;
    }

    return previsoes;
}

/*****************************************/
/*              Persistência             */
/*****************************************/

template <class T>
std::string OPFSupervisionado<T>::serializa(uchar flags)
{
    if (this->isPrecomputado)
        throw std::invalid_argument("Serialização para OPF pré-computado ainda não implementado");
    // Abre arquivo
    std::ostringstream saida(std::ios::out | std::ios::binary);

    int n_amostras = this->dadoDeTreinamento.linhas;
    int n_caracteristicas = this->dadoDeTreinamento.colunas;

    // Cabeçalho
    escreveBinario<char>(saida, "OPF", 3);
    escreveBinario<uchar>(saida, Tipo::Classificador);
    escreveBinario<uchar>(saida, flags);
    escreveBinario<uchar>(saida, static_cast<uchar>(0)); // Reserved flags byte
    escreveBinario<int>(saida, n_amostras);
    escreveBinario<int>(saida, n_caracteristicas);

    // Dado
    for (int i = 0; i < n_amostras; i++)
    {
        const T* dado = this->dadoDeTreinamento.linha(i);
        escreveBinario<T>(saida, dado, n_caracteristicas);
    }

    // Vértices
    for (int i = 0; i < n_amostras; i++)
    {
        escreveBinario<float>(saida, this->vertices[i].custo);
        escreveBinario<int>(saida, this->vertices[i].rotulo);
    }

    // Ordered_vertices
    escreveBinario<unsigned int>(saida, this->nodosOrdenados.data(), n_amostras);

    // Protótipos
    if (flags & SFlags::Supervisionado_SalvaPrototipos)
    {
        // Descobre quais são protótipos primeiro, porque precisamos da quantidade correta
        std::set<int> prots;
        for (int i = 0; i < n_amostras; i++)
        {
            if (this->vertices[i].isPrototipo)
                prots.insert(i);
        }

        escreveBinario<int>(saida, prots.size());
        for (auto it = prots.begin(); it != prots.end(); ++it)
            escreveBinario<int>(saida, *it);
    }

    return saida.str();
}

template <class T>
OPFSupervisionado<T> OPFSupervisionado<T>::desserializa(const std::string& conteudos)
{
    // Header
    int n_amostras;
    int n_caracteristicas;

    char cabecalho[4];

    OPFSupervisionado<float> opf;

    // Abre stream
    std::istringstream sistemaDeArquivosDeEntrada(conteudos); // , std::ios::in | std::ios::binary

    // Verifique se o fluxo é uma serialização do OPF
    lerbinario<char>(sistemaDeArquivosDeEntrada, cabecalho, 3);
    cabecalho[3] = '\0';
    if (strcmp(cabecalho, "OPF"))
        throw std::invalid_argument("A entrada não é uma serialização do OPF");

    // Obtém tipo and flags
    uchar tipo = lerBinario<uchar>(sistemaDeArquivosDeEntrada);
    uchar flags = lerBinario<uchar>(sistemaDeArquivosDeEntrada);
    lerBinario<uchar>(sistemaDeArquivosDeEntrada); // Byte reservado

    if (tipo != Tipo::Classificador)
        throw std::invalid_argument("A entrada não é uma serialização supervisionada da OPF");

    n_amostras = lerBinario<int>(sistemaDeArquivosDeEntrada);
    n_caracteristicas = lerBinario<int>(sistemaDeArquivosDeEntrada);

    // Dado
    int tamanho = n_amostras * n_caracteristicas;
    opf.dadoDeTreinamento = Mat<T>(n_amostras, n_caracteristicas);
    T* dado = opf.dadoDeTreinamento.linha(0);
    lerbinario<T>(sistemaDeArquivosDeEntrada, dado, tamanho);

    // Vértices
    opf.vertices = std::vector<Vertice>(n_amostras);
    for (int i = 0; i < n_amostras; i++)
    {
        opf.vertices[i].custo = lerBinario<float>(sistemaDeArquivosDeEntrada);
        opf.vertices[i].rotulo = lerBinario<int>(sistemaDeArquivosDeEntrada);
    }

    // Vértices ordenados
    opf.nodosOrdenados = std::vector<unsigned int>(n_amostras);
    lerbinario<unsigned int>(sistemaDeArquivosDeEntrada, opf.nodosOrdenados.data(), n_amostras);

    if (flags & SFlags::Supervisionado_SalvaPrototipos)
    {
        int prots = lerBinario<int>(sistemaDeArquivosDeEntrada);
        for (int i = 0; i < prots; i++)
        {
            int indice = lerBinario<int>(sistemaDeArquivosDeEntrada);
            opf.vertices[indice].isPrototipo = true;
        }
    }

    return std::move(opf);
}

/*****************************************/
/*              Acesso de dados          */
/*****************************************/

template <class T>
std::vector<std::vector<float>> OPFSupervisionado<T>::getPrototipos()
{
    std::set<int> prots;
    for (size_t i = 0; i < this->dadoDeTreinamento.linhas; i++)
    {
        if (this->vertices[i].isPrototipo)
            prots.insert(i);
    }

    std::vector<std::vector<float>> out(prots.size(), std::vector<float>(this->dadoDeTreinamento.colunas));
    int i = 0;
    for (auto it = prots.begin(); it != prots.end(); ++it, ++i)
    {
        for (int j = 0; j < this->dadoDeTreinamento.colunas; j++)
        {
            out[i][j] = this->dadoDeTreinamento[*it][j];
        }
    }

    return out;
}

/*****************************************/

/******** OPF Não supervisionado  ********/

// Índice + distância até o outro vértice
using Pdist = std::pair<int, float>;

static bool comparaVizinho(const Pdist& estruturaHeapEsquerda, const Pdist& estruturaHeapDireita)
{
    return estruturaHeapEsquerda.second < estruturaHeapDireita.second;
}

// Classe auxiliar para encontrar os k-vizinhos mais próximos de um determinado nó
class MelhorKVizinho
{
private:
    int k;
    std::vector<Pdist> heap; // índice, distância

public:
    // Empty initializer
    MelhorKVizinho(int k) : k(k) {this->heap.reserve(k);}
    // Tries to insert another element to the heap
    void insert(int indice, float distancia)
    {
        if (heap.size() < static_cast<unsigned int>(this->k))
        {
            heap.push_back(Pdist(indice, distancia));
            push_heap(this->heap.begin(), this->heap.end(), comparaVizinho);
        }
        else
        {
            // Se o novo ponto está mais perto do que o vizinho mais distante
            Pdist maisDistante = this->heap.front();
            if (distancia < maisDistante.second)
            {
                // Remove um do heap e adicione o outro
                pop_heap(this->heap.begin(), this->heap.end(), comparaVizinho);
                this->heap[this->k-1] = Pdist(indice, distancia);
                push_heap(this->heap.begin(), this->heap.end(), comparaVizinho);
            }
        }
    }

    std::vector<Pdist>& get_knn() { return heap; }
};


/**
 * Plain class to store vértice information
 */
class VerticeKNN
{
public:
    VerticeKNN()
    {
        this->pred = -1;
    }

    std::set<Pdist> adj; // Adjacência do vértice
    size_t index;        // Índice na lista - facilita as pesquisas
    int label;           // Rotulo atribuído
    int pred;            // Vértice predecessor
    float value;         // Valor do caminho
    float rho;           // função de densidade de probabilidade
};

// Classificador OPF não supervisionado
template <class T=float>
class OPFNaoSupervisionado
{
private:
    // Model
    std::shared_ptr<const Mat<T>> dadoDeTreinamento;   // Dado de treinamento (Vetores originais ou matriz de distância)
    funcaoDistancia<T> distancia;       // Função de distância
    std::vector<VerticeKNN> vertices;   // Modelo aprendido
    std::vector<int> fila;              // Fila de prioridades implementada como uma pesquisa linear em um vetor
    int k;                              // O número de vizinhos para construir o gráfico
    int n_clusters;                     // Número de clusters no modelo -- computado durante o ajuste

    // Atributos de treinamento
    float sigma_sq;             // Sigma ao quadrado, usado para calcular função de distribuição de probabilidades
    float delta;                // Termo de ajuste
    float denominador;          // sqrt(2 * math.pi * sigma_sq) -- computado apenas uma vez

    // Options
    float limiar;
    bool anomalia;
    bool isPrecomputado;

    // Recursos da fila
    int get_max();

    // Subrotinas de treinamento
    void constroiGrafo();
    void inicializaGrafo();
    void cluster();

public:
    OPFNaoSupervisionado(int k=5, bool anomalia=false, float limiar=.1, bool isPrecomputado=false, funcaoDistancia<T> distancia=distanciaEuclidiana<T>);

    void ajusta(const Mat<T> &dadoDeTreinamento);
    std::vector<int> ajustaPrevisao(const Mat<T> &dadoDeTreinamento);
    std::vector<int> prediz(const Mat<T> &dadoDeTeste);

    void buscaMelhorK(Mat<float>& dadoDeTreinamento, int kmin, int kmax, int etapa= 1, bool isPrecomputa= true);

    // Informação do cluster
    float metricaDeQualidade();

    // Getters & Setters
    int get_n_clusters() {return this->n_clusters;}
    int get_k() {return this->k;}
    bool getAnomalia() {return this->anomalia;}
    float getLimiar() {return this->limiar;}
    void setLimiar(float limiar) { this->limiar = limiar;}
    bool getPrecomputado() {return this->isPrecomputado;}

    // Funções de serialização
    std::string serializa(uchar flags= 0);
    static OPFNaoSupervisionado<T> desserializa(const std::string& contents);
};

template <class T>
OPFNaoSupervisionado<T>::OPFNaoSupervisionado(int k, bool anomalia, float limiar, bool isPrecomputado, funcaoDistancia<T> distancia)
{
    this->k = k;
    this->isPrecomputado = isPrecomputado;
    this->anomalia = anomalia;
    if (this->anomalia)
        this->n_clusters = 2;
    this->limiar = limiar;
    this->distancia = distancia;
}

// Constrói o grafo KNN
template <class T>
void OPFNaoSupervisionado<T>::constroiGrafo()
{
    this->sigma_sq = 0.;

    // Proporcional ao comprimento da maior aresta
    for (size_t i = 0; i < this->vertices.size(); i++)
    {
        // Encontre os k-vizinhos mais próximos
        MelhorKVizinho bk(this->k);
        for (size_t j = 0; j < this->vertices.size(); j++)
        {
            if (i != j)
            {
                float distancia;
                if (this->isPrecomputado)
                    distancia = this->dadoDeTreinamento->em(i, j);
                else
                    distancia = this->distancia(this->dadoDeTreinamento->linha(i), this->dadoDeTreinamento->linha(j), this->dadoDeTreinamento->colunas);

                bk.insert(j, distancia);
            }
        }

        std::vector<Pdist> knn = bk.get_knn();
        for (auto it = knn.cbegin(); it != knn.cend(); ++it)
        {
            // Uma vez que o gráfico não é direcionado, faça conexões de ambos os vertices
            this->vertices[i].adj.insert(*it);
            this->vertices[it->first].adj.insert(Pdist(i, it->second));

            // Encontrando sigma
            if (it->second > this->sigma_sq)
                this->sigma_sq = it->second;
        }
    }

    this->sigma_sq /= 3;
    this->sigma_sq = 2 * (this->sigma_sq * this->sigma_sq);
    this->denominador = sqrt(2 * M_PI * this->sigma_sq);
}

// Inicializa os vertices do grafo
template <class T>
void OPFNaoSupervisionado<T>::inicializaGrafo()
{
    // Computa rho
    std::set<Pdist>::iterator it;
    for (size_t i = 0; i < this->vertices.size(); i++)
    {
        int n_vizinhos = this->vertices[i].adj.size(); // Um vértice pode ter mais de k vizinhos
        float div = this->denominador * n_vizinhos;
        float sum = 0;

        for (it = this->vertices[i].adj.cbegin(); it != this->vertices[i].adj.cend(); ++it)
        {
            float distancia = it->second;
            sum += expf((-distancia * distancia) / this->sigma_sq);
        }

        this->vertices[i].rho = sum / div;
    }

    // Computa delta
    this->delta = INF;
    for (size_t i = 0; i < this->vertices.size(); i++)
    {
        for (it = this->vertices[i].adj.begin(); it != this->vertices[i].adj.end(); ++it)
        {
            float diff = abs(this->vertices[i].rho - this->vertices[it->first].rho);
            if (diff != 0 && this->delta > diff)
                this->delta = diff;
        }
    }

    // E, finalmente, inicializa cada vértice
    this->fila.resize(this->vertices.size());
    for (size_t i = 0; i < this->vertices.size(); i++)
    {
        this->vertices[i].value = this->vertices[i].rho - this->delta;
        this->fila[i] = static_cast<int>(i);
    }
}

// Obtém o vértice com o maior valor de caminho
template <class T>
int OPFNaoSupervisionado<T>::get_max()
{
    float valorMaximo = -INF;
    int indiceMaximo = -1;
    int size = this->fila.size();
    for (int i = 0; i < size; i++)
    {
        int indice = this->fila[i];
        if (this->vertices[indice].value > valorMaximo)
        {
            indiceMaximo = i;
            valorMaximo = this->vertices[indice].value;
        }
    }

    int melhor = this->fila[indiceMaximo];
    int tmp = this->fila[size-1];
    this->fila[size-1] = this->fila[indiceMaximo];
    this->fila[indiceMaximo] = tmp;
    this->fila.pop_back();

    return melhor;
}

// Agrupamento com OPF
template <class T>
void OPFNaoSupervisionado<T>::cluster()
{
    // Rótulo do cluster
    int rotulo = 0;
    // Fila de prioridade
    while (!this->fila.empty())
    {
        int s = this->get_max(); // Extrai o valor mais alto

        // Se não tem nenhum antecessor, faz dele um protótipo
        if (this->vertices[s].pred == -1)
        {
            this->vertices[s].label = rotulo++;
            this->vertices[s].value = this->vertices[s].rho;
        }

        // Itera e conquista seus vizinhos
        for (auto it = this->vertices[s].adj.begin(); it != this->vertices[s].adj.end(); ++it)
        {
            int t = it->first;
            if (this->vertices[t].value < this->vertices[s].value)
            {
                float rho = std::min(this->vertices[s].value, this->vertices[t].rho);
                
                if (rho > this->vertices[t].value)
                {
                    this->vertices[t].label = this->vertices[s].label;
                    this->vertices[t].pred = s;
                    this->vertices[t].value = rho;
                }
            }
        }
    }

    this->n_clusters = rotulo;
}


// Ajusta o modelo
template <class T>
void OPFNaoSupervisionado<T>::ajusta(const Mat<T> &dadoDeTreinamento)
{
    this->dadoDeTreinamento = std::shared_ptr<const Mat<T>>(&dadoDeTreinamento, [](const Mat<T> *p) {});
    this->vertices = std::vector<VerticeKNN>(this->dadoDeTreinamento->linhas);
    this->constroiGrafo();
    this->inicializaGrafo();
    if (!this->anomalia)
        this->cluster();
}

// Ajusta e prevê para todos os vertices
template <class T>
std::vector<int> OPFNaoSupervisionado<T>::ajustaPrevisao(const Mat<T> &dadoDeTreinamento)
{
    this->ajusta(dadoDeTreinamento);

    std::vector<int> rotulos(this->vertices.size());

    if (this->anomalia)
        for (size_t i = 0; i < this->vertices.size(); i++)
            rotulos[i] = (this->vertices[i].rho < this->limiar) ? 1 : 0;
    else
        for (size_t i = 0; i < this->vertices.size(); i++)
            rotulos[i] = this->vertices[i].label;

    return rotulos;
}

// Prevê a pertinência do cluster
template <class T>
std::vector<int> OPFNaoSupervisionado<T>::prediz(const Mat<T> &dadoDeTeste)
{
    std::vector<int> preds(dadoDeTeste.linhas);
    // Para cada amostra de teste
    for (size_t i = 0; i < dadoDeTeste.linhas; i++)
    {
        // Encontra os vizinhos mais próximos
        MelhorKVizinho bk(this->k);
        for (size_t j = 0; j < this->vertices.size(); j++)
        {
            if (i != j)
            {
                float distancia;
                if (this->isPrecomputado)
                    distancia = dadoDeTeste.em(i, j);
                else
                    distancia = this->distancia(dadoDeTeste[i], this->dadoDeTreinamento->linha(j), this->dadoDeTreinamento->colunas);

                bk.insert(j, distancia);
            }
        }

        // Computa o rho de teste
        std::vector<Pdist> vizinhos = bk.get_knn();
        int n_vizinhos = vizinhos.size();

        float div = this->denominador * n_vizinhos;
        float sum = 0;

        for (int j = 0; j < n_vizinhos; j++)
        {
            float distancia = vizinhos[j].second;
            sum += expf((-distancia * distancia) / this->sigma_sq);
        }

        float rho = sum / div;

        if (this->anomalia)
        {
            // E retorna a detecção de anomalia com base na densidade do gráfico
            preds[i] = (rho < this->limiar) ? 1 : 0;
        }
        else
        {
            // E descobre qual vértice conquista esta amostra de teste
            float valorMaximo = -INF;
            int indiceMaximo = -1;

            for (int j = 0; j < n_vizinhos; j++)
            {
                int s = vizinhos[j].first;  // indice, distancia
                float val = std::min(this->vertices[s].value, rho);
                if (val > valorMaximo)
                {
                    valorMaximo = val;
                    indiceMaximo = s;
                }
            }

            preds[i] = this->vertices[indiceMaximo].label;
        }
    }

    return preds;
}

// Métrica de qualidade
// Fonte: A Robust Extension of the Mean Shift Algorithm using Optimum Path Forest
// Leonardo Rocha, Alexandre Falcao, and Luis Meloni
template <class T>
float OPFNaoSupervisionado<T>::metricaDeQualidade()
{
    if (this->anomalia)
        throw std::invalid_argument("Métrica de qualidade ainda não implementada para detecção de anomalias");
    std::vector<float> w(this->n_clusters, 0);
    std::vector<float> w_(this->n_clusters, 0);
    for (size_t i = 0; i < this->dadoDeTreinamento->linhas; i++)
    {
        int l = this->vertices[i].label;

        for (auto it = this->vertices[i].adj.begin(); it != this->vertices[i].adj.end(); ++it)
        {
            int l_ = this->vertices[it->first].label;
            float tmp = 0;

            if (it->second != 0)
                tmp = 1. / it->second;

            if (l == l_)
                w[l] += tmp;
            else
                w_[l] += tmp;
        }
    }

    float C = 0;
    for (int i = 0; i < this->n_clusters; i++)
        C += w_[i] / (w_[i] + w[i]);

    return C;
}

// Método de força bruta para encontrar o melhor valor de k
template <class T>
void OPFNaoSupervisionado<T>::buscaMelhorK(Mat<float>& dadoDeTreinamento, int kmin, int kmax, int etapa, bool isPrecomputa)
{
    std::cout << "isPrecomputa " << isPrecomputa << std::endl;
    float melhorQualidade = INF;
    OPFNaoSupervisionado<float> best_opf;
    MatrizDeDistancias<float> distancias;
    if (isPrecomputa)
        distancias = MatrizDeDistancias<float>(dadoDeTreinamento, this->distancia);

    for (int k = kmin; k <= kmax; k += etapa)
    {
        // Instancia e treina o modelo
        OPFNaoSupervisionado<float> opf(k, false, 0, isPrecomputa, this->distancia);
        if (isPrecomputa)
            opf.ajusta(distancias);
        else
            opf.ajusta(dadoDeTreinamento);

        std::cout << k << ": " << opf.n_clusters << std::endl;
        // Compara seu grau de agrupamento
        float qualidade = opf.metricaDeQualidade(); // Normalized cut
        if (qualidade < melhorQualidade)
        {
            melhorQualidade = qualidade;
            best_opf = opf;
        }

    }

    if (melhorQualidade == INF)
    {
        std::ostringstream ss;
        ss  << "Nenhuma busca com kmin " << kmin << ", kmax " << kmax << ", e etapa " << etapa
            << ". Os argumentos podem estar fora de ordem.";
        std::cerr << ss.str() << std::endl;
        throw 0;
    }


    if (this->isPrecomputado)
        this->dadoDeTreinamento = std::shared_ptr<Mat<T>>(&distancias, std::default_delete<Mat<T>>());
    else
        this->dadoDeTreinamento = std::shared_ptr<Mat<T>>(&dadoDeTreinamento, [](Mat<T> *p) {});

    this->k = best_opf.k;
    this->n_clusters = best_opf.n_clusters;
    this->vertices = best_opf.vertices;
    this->denominador = best_opf.denominador;
    this->sigma_sq = best_opf.sigma_sq;
    this->delta = best_opf.delta;
}

/*****************************************/
/*              Persistência             */
/*****************************************/

template <class T>
std::string OPFNaoSupervisionado<T>::serializa(uchar flags)
{
    if (this->isPrecomputado)
        throw std::invalid_argument("Serialização para OPF Pré-computado ainda não implementado");

    // Abre Arquivo
    std::ostringstream saida;
    int n_amostras = this->dadoDeTreinamento->linhas;
    int n_caracteristicas = this->dadoDeTreinamento->colunas;

    // Flags de saída
    flags = 0; // Por enquanto, não há bandeiras definidas pelo usuário
    if (this->anomalia)
        flags += SFlags::NaoSupervisionado_Anomalia;

    // Cabeçalho
    escreveBinario<char>(saida, "OPF", 3);
    escreveBinario<uchar>(saida, Tipo::Agrupamento);
    escreveBinario<uchar>(saida, flags);
    escreveBinario<uchar>(saida, static_cast<uchar>(0)); // Byte reservado
    escreveBinario<int>(saida, n_amostras);
    escreveBinario<int>(saida, n_caracteristicas);

    // Dado escalar
    escreveBinario<int>(saida, this->k);
    if (!this->anomalia)
        escreveBinario<int>(saida, this->n_clusters);
    else
        escreveBinario<float>(saida, this->limiar);
    escreveBinario<float>(saida, this->denominador);
    escreveBinario<float>(saida, this->sigma_sq);

    // Dado
    for (int i = 0; i < n_amostras; i++)
    {
        const T* dado = this->dadoDeTreinamento->linha(i);
        escreveBinario<T>(saida, dado, n_caracteristicas);
    }

    // Vértices
    for (int i = 0; i < n_amostras; i++)
    {
        escreveBinario<float>(saida, this->vertices[i].value);
        if (!this->anomalia)
            escreveBinario<int>(saida, this->vertices[i].label);
    }

    return saida.str();
}

template <class T>
OPFNaoSupervisionado<T> OPFNaoSupervisionado<T>::desserializa(const std::string& contents)
{
    OPFNaoSupervisionado<float> opf;

    // Abre stream
    std::istringstream ifs(contents);

    /// Cabeçalho
    int n_amostras;
    int n_caracteristicas;
    char header[4];

    // Verifique se o stream é uma serialização de OPF
    lerbinario<char>(ifs, header, 3);
    header[3] = '\0';
    if (strcmp(header, "OPF"))
        throw std::invalid_argument("Input is not an OPF serialization");

    // Obtém tipo e flags
    uchar tipo = lerBinario<uchar>(ifs);
    uchar flags = lerBinario<uchar>(ifs); // Flags byte
    lerBinario<uchar>(ifs); // Byte reservado

    if (flags & SFlags::NaoSupervisionado_Anomalia)
        opf.anomalia = true;

    if (tipo != Tipo::Agrupamento)
        throw std::invalid_argument("A entrada não é uma serialização OPF não supervisionada");

    // Tamanho do dado
    n_amostras = lerBinario<int>(ifs);
    n_caracteristicas = lerBinario<int>(ifs);

    // Dado escalar
    opf.k = lerBinario<int>(ifs);
    if (!opf.anomalia)
        opf.n_clusters = lerBinario<int>(ifs);
    else
    {
        opf.limiar = lerBinario<float>(ifs);
        opf.n_clusters = 2;
    }
    opf.denominador = lerBinario<float>(ifs);
    opf.sigma_sq = lerBinario<float>(ifs);

    /// Data
    // Variável temporária para ler dado, já que dadoDeTreinamento da OPF é const
    auto dadoDeTreinamento = std::shared_ptr<Mat<T>>(new Mat<T>(n_amostras, n_caracteristicas), std::default_delete<Mat<T>>());
    // Lê dado
    int tamanho = n_amostras * n_caracteristicas;
    T* dado = dadoDeTreinamento->linha(0);
    lerbinario<T>(ifs, dado, tamanho);
    // Atribui ao OPF
    opf.dadoDeTreinamento = dadoDeTreinamento;

    // Vértices
    opf.vertices = std::vector<VerticeKNN>(n_amostras);
    for (int i = 0; i < n_amostras; i++)
    {
        opf.vertices[i].value = lerBinario<float>(ifs);
        if (!opf.anomalia)
            opf.vertices[i].label = lerBinario<int>(ifs);
    }

    return std::move(opf);
}

/*****************************************/

}

#endif
