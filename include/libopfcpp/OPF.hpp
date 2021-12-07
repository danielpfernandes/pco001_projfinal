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
class DistMat: public Mat<T>
{
private:
    T diag_vals = static_cast<T>(0);
    int buscaIndice(int i, int j) const;
public:
    DistMat(){ this->linhas = this->colunas = this->tamanho = 0;};
    DistMat(const DistMat& outro);
    DistMat(const Mat<T>& caracteristicas, funcaoDistancia<T> distancia=distanciaEuclidiana<T>);
    virtual T& em(size_t i, size_t j);
    const virtual T at(size_t i, size_t j) const;
};

// A primeira linha tem n-1 colunas, a segunda tem n-2, e assim por diante até linha n tem 0 colunas.
// Assim
#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))
template <class T>
inline int DistMat<T>::buscaIndice(int i, int j) const
{
    if (i > j)
        SWAP(i, j);
    return ((((this->linhas << 1) - i - 1) * i) >> 1) + (j - i - 1);
}

template <class T>
DistMat<T>::DistMat(const DistMat& outro)
{
    this->linhas = outro.linhas;
    this->colunas = outro.colunas;
    this->tamanho = outro.tamanho;
    this->dado = outro.dado;
}

template <class T>
DistMat<T>::DistMat(const Mat<T>& caracteristicas, funcaoDistancia<T> distancia)
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
T& DistMat<T>::em(size_t i, size_t j)
{
    if (i == j)
        return this->diag_vals = static_cast<T>(0);
    return this->dado.get()[this->buscaIndice(i, j)];
}

template <class T>
const T DistMat<T>::at(size_t i, size_t j) const
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
    BRANCO, // Novo nó
    CINZA,  // Na heap
    PRETO  // Já visto
};

/**
 * Classe simples para armazenar informações de nó
 */
class Nodo
{
public:
    Nodo()
    {
        this->cor = BRANCO;
        this->predecessor = -1;
        this->custo = INF;
        this->isPrototipo = false;
    }

    size_t indice;      // Índice na lista - facilita as pesquisas
    Color cor;          // Cor na heap. branco: nunca visitada, cinza: na heap, preto: removido da heap
    float custo;        // Custo para alcançar o nó
    int rotuloVerdade;  // Valor de referência
    int rotulo;         // Rótulo atribuído
    int predecessor;    // Nó antecessor
    bool isPrototipo;   // Se o nó é um protótipo
};

/**
 * Estrutura de dados de heap para usar como uma fila de prioridade
 *
 */
class heap
{
private:
    std::vector<Nodo> *nodos; // Uma referência para o vetor de contêiner original
    std::vector<Nodo*> vec;   // Um vetor de ponteiros para construir a heap em cima

    static bool comparaElemento(const Nodo* estruturaHeapEsquerda, const Nodo* estruturaHeapDireita)
    {
        return estruturaHeapEsquerda->custo >= estruturaHeapDireita->custo;
    }

public:
    // Construtor de tamanho
    heap(std::vector<Nodo> *nodos, const std::vector<int> &rotulos)
    {
        this->nodos = nodos;
        size_t n = nodos->size();
        this->vec.reserve(n);
        for (size_t i = 0; i < n; i++)
        {
            (*this->nodos)[i].indice = i;
            (*this->nodos)[i].rotuloVerdade = (*this->nodos)[i].rotulo = rotulos[i];
        }
    }
    /**
     * Insere o novo elemento na heap
     * @param item elemento
     * @param custo custo
     */
    void push(int item, float custo)
    {
        // Atualiza o valor de custo do nó
        (*this->nodos)[item].custo = custo;

        // Já presente na heap
        if ((*this->nodos)[item].cor == CINZA)
            make_heap(this->vec.begin(), this->vec.end(), comparaElemento); // Remake the heap

        // Novo na heap
        else if ((*this->nodos)[item].cor == BRANCO)
        {
            (*this->nodos)[item].cor = CINZA;
            this->vec.push_back(&(*this->nodos)[item]);
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
        // Atualiza o valor de custo do nó
        (*this->nodos)[item].custo = custo;
        if ((*this->nodos)[item].cor == BRANCO)
        {
            (*this->nodos)[item].cor = CINZA;
            this->vec.push_back(&(*this->nodos)[item]);
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
        Nodo *frente = this->vec.front();
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
    std::vector<Nodo> nodos;  // Modelo aprendido

    // Lista de nós ordenados pelo custo. Útil para acelerar a classificação
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
 * - O primeiro passo no procedimento de treinamento da OPF. Encontra o nós protótipos
 *   usando o algoritmo da Árvore Geradora Mínima de Prim.
 * - Qualquer nó com um nó adjacente de uma classe diferente é tomado como um protótipo.
 * @tparam T
 * @param rotulos Rótulos dos dados
 */
template <class T>
void OPFSupervisionado<T>::prototipoDePrim(const std::vector<int> &rotulos)
{
    this->nodos = std::vector<Nodo>(this->dadoDeTreinamento.linhas);
    heap cabeca(&this->nodos, rotulos); // heap como uma fila de prioridades

    // Primeiro nó arbitrário
    cabeca.push(0, 0);

    while(!cabeca.isVazia())
    {
        // Pega a cabeça da heap e marca-a preta
        size_t s = cabeca.pop();

        // Definição de protótipo
        int pred = this->nodos[s].predecessor;
        if (pred != NIL)
        {
            // Encontra pontos na fronteira entre duas classes...
            if (this->nodos[s].rotuloVerdade != this->nodos[pred].rotuloVerdade)
            {
                // E os define como protótipos
                this->nodos[s].isPrototipo = true;
                this->nodos[pred].isPrototipo = true;
            }
        }


        // Seleção de arestas
        #pragma omp parallel for default(shared)
        for (size_t t = 0; t < this->nodos.size(); t++)
        {
            // Se os nós são diferentes e t não foi retirado da heap (marcado preto)
            if (s != t && this->nodos[t].cor != PRETO)
            {
                // Calcula o peso
                float peso;
                if (this->isPrecomputado)
                    peso = this->dadoDeTreinamento[s][t];
                else
                    peso = this->distancia(this->dadoDeTreinamento[s], this->dadoDeTreinamento[t], this->dadoDeTreinamento.colunas);

                // Atribui se menor do que o valor atual
                if (peso < this->nodos[t].custo)
                {
                    this->nodos[t].predecessor = static_cast<int>(s);
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
    heap cabeca(&this->nodos, rotulos); // heap como uma fila de prioridade

    // Inicialização
    for (Nodo& nodo: this->nodos)
    {
        nodo.cor = BRANCO;
        // Protótipos custo 0, não ter antecessor e povoar o heap
        if (nodo.isPrototipo)
        {
            nodo.predecessor = NIL;
            nodo.custo = 0;
        }
        else // Outros nós iniciam com custo = INF
        {
            nodo.custo = INF;
        }
        // Uma vez que todos os nós estão conectados a todos os outros
        cabeca.push(nodo.indice, nodo.custo);
    }

    // Lista de nós ordenados por custo
    // Útil para acelerar a classificação
    this->nodosOrdenados.reserve(this->nodos.size());

    // consome a fila
    while(!cabeca.isVazia())
    {
        int s = cabeca.pop();
        this->nodosOrdenados.push_back(s);

        // Itera sobre todos os vizinhos
        #pragma omp parallel for default(shared)
        for (int t = 0; t < (int) this->nodos.size(); t++)
        {
            if (s != t && this->nodos[s].custo < this->nodos[t].custo)
            {
                // Computa o peso
                float peso;
                if (isPrecomputado)
                    peso = this->dadoDeTreinamento[s][t];
                else
                    peso = distancia(this->dadoDeTreinamento[s], this->dadoDeTreinamento[t], this->dadoDeTreinamento.colunas);

                float cost = std::max(peso, this->nodos[s].custo);
                if (cost < this->nodos[t].custo)
                {
                    this->nodos[t].predecessor = s;
                    this->nodos[t].rotulo = this->nodos[s].rotuloVerdade;

                    #pragma omp critical(updateHeap)
                    cabeca.atualizaCusto(t, cost);
                }
            }
        }
        cabeca.heapify();
    }
}

/**
 * Classify a set of samples using a model trained by OPFSupervisionado::ajusta.
 *
 * Inputs:
 *  - dadoDeTeste:
 *    - original feature vectors [n_test_samples, n_features]      -- if isPrecomputado == false
 *    - distancia matrix          [n_test_samples, n_train_samples] -- if isPrecomputado == true
 *
 * Returns:
 *  - predictions:
 *    - a vector<int> of tamanho [n_test_samples] with classification outputs.
 */
template <class T>
std::vector<int> OPFSupervisionado<T>::prediz(const Mat<T> &dadoDeTeste)
{
    int n_test_samples = (int) dadoDeTeste.linhas;
    int n_train_samples = (int) this->nodos.size();

    // Output predictions
    std::vector<int> predictions(n_test_samples);

    #pragma omp parallel for default(shared)
    for (int i = 0; i < n_test_samples; i++)
    {
        int idx = this->nodosOrdenados[0];
        int min_idx = 0;
        T min_cost = INF;
        T weight = 0;

        // 'nodosOrdenados' contains sample indices ordered by custo, so if the current
        // best connection costs less than the next node, it is useless to keep looking.
        for (int j = 0; j < n_train_samples && min_cost > this->nodos[idx].custo; j++)
        {
            // Get the next node in the ordered list
            idx = this->nodosOrdenados[j];

            // Compute its distancia to the query point
            if (isPrecomputado)
                weight = dadoDeTeste[i][idx];
            else
                weight = distancia(dadoDeTeste[i], this->dadoDeTreinamento[idx], this->dadoDeTreinamento.colunas);

            // The custo corresponds to the max between the distancia and the reference custo
            float cost = std::max(weight, this->nodos[idx].custo);

            if (cost < min_cost)
            {
                min_cost = cost;
                min_idx = idx;
            }
        }

        predictions[i] = this->nodos[min_idx].rotulo;
    }

    return predictions;
}

/*****************************************/
/*              Persistence              */
/*****************************************/

template <class T>
std::string OPFSupervisionado<T>::serializa(uchar flags)
{
    if (this->isPrecomputado)
        throw std::invalid_argument("Serialization for isPrecomputado OPF not implemented yet");
    // Open file
    std::ostringstream output(std::ios::out | std::ios::binary);

    int n_samples = this->dadoDeTreinamento.linhas;
    int n_features = this->dadoDeTreinamento.colunas;

    // Header
    escreveBinario<char>(output, "OPF", 3);
    escreveBinario<uchar>(output, Tipo::Classificador);
    escreveBinario<uchar>(output, flags);
    escreveBinario<uchar>(output, static_cast<uchar>(0)); // Reserved flags byte
    escreveBinario<int>(output, n_samples);
    escreveBinario<int>(output, n_features);

    // Data
    for (int i = 0; i < n_samples; i++)
    {
        const T* data = this->dadoDeTreinamento.linha(i);
        escreveBinario<T>(output, data, n_features);
    }

    // Nodes
    for (int i = 0; i < n_samples; i++)
    {
        escreveBinario<float>(output, this->nodos[i].custo);
        escreveBinario<int>(output, this->nodos[i].rotulo);
    }

    // Ordered_nodes
    escreveBinario<unsigned int>(output, this->nodosOrdenados.data(), n_samples);

    // Prototypes
    if (flags & SFlags::Supervisionado_SalvaPrototipos)
    {
        // Find which are prototypes first, because we need the correct amount
        std::set<int> prots;
        for (int i = 0; i < n_samples; i++)
        {
            if (this->nodos[i].isPrototipo)
                prots.insert(i);
        }

        escreveBinario<int>(output, prots.size());
        for (auto it = prots.begin(); it != prots.end(); ++it)
            escreveBinario<int>(output, *it);
    }

    return output.str();
}

template <class T>
OPFSupervisionado<T> OPFSupervisionado<T>::desserializa(const std::string& conteudos)
{
    // Header
    int n_samples;
    int n_features;

    char header[4];

    OPFSupervisionado<float> opf;

    // Open stream
    std::istringstream ifs(conteudos); // , std::ios::in | std::ios::binary

    // Check if stream is an OPF serialization
    lerbinario<char>(ifs, header, 3);
    header[3] = '\0';
    if (strcmp(header, "OPF"))
        throw std::invalid_argument("Input is not an OPF serialization");

    // Get type and flags
    uchar type = lerBinario<uchar>(ifs);
    uchar flags = lerBinario<uchar>(ifs);
    lerBinario<uchar>(ifs); // Reserved byte

    if (type != Tipo::Classificador)
        throw std::invalid_argument("Input is not a Supervised OPF serialization");

    n_samples = lerBinario<int>(ifs);
    n_features = lerBinario<int>(ifs);

    // Data
    int size = n_samples * n_features;
    opf.dadoDeTreinamento = Mat<T>(n_samples, n_features);
    T* data = opf.dadoDeTreinamento.linha(0);
    lerbinario<T>(ifs, data, size);

    // Nodes
    opf.nodos = std::vector<Nodo>(n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        opf.nodos[i].custo = lerBinario<float>(ifs);
        opf.nodos[i].rotulo = lerBinario<int>(ifs);
    }

    // Ordered_nodes
    opf.nodosOrdenados = std::vector<unsigned int>(n_samples);
    lerbinario<unsigned int>(ifs, opf.nodosOrdenados.data(), n_samples);

    if (flags & SFlags::Supervisionado_SalvaPrototipos)
    {
        int prots = lerBinario<int>(ifs);
        for (int i = 0; i < prots; i++)
        {
            int idx = lerBinario<int>(ifs);
            opf.nodos[idx].isPrototipo = true;
        }
    }

    return std::move(opf);
}

/*****************************************/
/*              Data Access              */
/*****************************************/

template <class T>
std::vector<std::vector<float>> OPFSupervisionado<T>::getPrototipos()
{
    std::set<int> prots;
    for (size_t i = 0; i < this->dadoDeTreinamento.linhas; i++)
    {
        if (this->nodos[i].isPrototipo)
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

/******** Unsupervised ********/

// Index + distancia to another node
using Pdist = std::pair<int, float>;

static bool compare_neighbor(const Pdist& lhs, const Pdist& rhs)
{
    return lhs.second < rhs.second;
}

// Aux class to find the k nearest neighbors from a given node
// In the future, this should be replaced by a kdtree
class BestK
{
private:
    int k;
    std::vector<Pdist> heap; // idx, dist

public:
    // Empty initializer
    BestK(int k) : k(k) {this->heap.reserve(k);}
    // Tries to insert another element to the heap
    void insert(int idx, float dist)
    {
        if (heap.size() < static_cast<unsigned int>(this->k))
        {
            heap.push_back(Pdist(idx, dist));
            push_heap(this->heap.begin(), this->heap.end(), compare_neighbor);
        }
        else
        {
            // If the new point is closer than the farthest neighbor
            Pdist farthest = this->heap.front();
            if (dist < farthest.second)
            {
                // Remove one from the heap and add the other
                pop_heap(this->heap.begin(), this->heap.end(), compare_neighbor);
                this->heap[this->k-1] = Pdist(idx, dist);
                push_heap(this->heap.begin(), this->heap.end(), compare_neighbor);
            }
        }
    }

    std::vector<Pdist>& get_knn() { return heap; }
};


/**
 * Plain class to store node information
 */
class NodeKNN
{
public:
    NodeKNN()
    {
        this->pred = -1;
    }

    std::set<Pdist> adj; // Nodo adjacency
    size_t index;        // Index on the list -- makes searches easier
    int label;           // Assigned rotulo
    int pred;            // Predecessor node
    float value;         // Path value
    float rho;           // probability density function
};

// Unsupervised OPF classifier
template <class T=float>
class UnsupervisedOPF
{
private:
    // Model
    std::shared_ptr<const Mat<T>> train_data;   // Training dado (original vectors or distancia matrix)
    funcaoDistancia<T> distance; // Distance function
    std::vector<NodeKNN> nodes;    // Learned model
    std::vector<int> queue;        // Priority queue implemented as a linear search in a vector
    int k;                         // The number of neighbors to build the graph
    int n_clusters;                // Number of clusters in the model -- computed during ajusta

    // Training attributes
    float sigma_sq;             // Sigma squared, used to compute probability distribution function
    float delta;                // Adjustment term
    float denominator;          // sqrt(2 * math.pi * sigma_sq) -- compute it only once

    // Options
    float thresh;
    bool anomaly;
    bool precomputed;

    // Queue capabilities
    int get_max();

    // Training subroutines
    void build_graph();
    void build_initialize();
    void cluster();

public:
    UnsupervisedOPF(int k=5, bool anomaly=false, float thresh=.1, bool precomputed=false, funcaoDistancia<T> distance=distanciaEuclidiana<T>);

    void fit(const Mat<T> &train_data);
    std::vector<int> fit_predict(const Mat<T> &train_data);
    std::vector<int> predict(const Mat<T> &test_data);

    void find_best_k(Mat<float>& train_data, int kmin, int kmax, int step=1, bool precompute=true);

    // Agrupamento info
    float quality_metric();

    // Getters & Setters
    int get_n_clusters() {return this->n_clusters;}
    int get_k() {return this->k;}
    bool get_anomaly() {return this->anomaly;}
    float get_thresh() {return this->thresh;}
    void set_thresh(float thresh) {this->thresh = thresh;}
    bool get_precomputed() {return this->precomputed;}

    // Serialization functions
    std::string serialize(uchar flags=0);
    static UnsupervisedOPF<T> unserialize(const std::string& contents);
};

template <class T>
UnsupervisedOPF<T>::UnsupervisedOPF(int k, bool anomaly, float thresh, bool precomputed, funcaoDistancia<T> distance)
{
    this->k = k;
    this->precomputed = precomputed;
    this->anomaly = anomaly;
    if (this->anomaly)
        this->n_clusters = 2;
    this->thresh = thresh;
    this->distance = distance;
}

// Builds the KNN graph
template <class T>
void UnsupervisedOPF<T>::build_graph()
{
    this->sigma_sq = 0.;

    // Proportional to the length of the biggest edge
    for (size_t i = 0; i < this->nodes.size(); i++)
    {
        // Find the k nearest neighbors
        BestK bk(this->k);
        for (size_t j = 0; j < this->nodes.size(); j++)
        {
            if (i != j)
            {
                float dist;
                if (this->precomputed)
                    dist = this->train_data->em(i, j);
                else
                    dist = this->distance(this->train_data->linha(i), this->train_data->linha(j), this->train_data->colunas);

                bk.insert(j, dist);
            }
        }

        std::vector<Pdist> knn = bk.get_knn();
        for (auto it = knn.cbegin(); it != knn.cend(); ++it)
        {
            // Since the graph is undirected, make connections from both nodos
            this->nodes[i].adj.insert(*it);
            this->nodes[it->first].adj.insert(Pdist(i, it->second));

            // Finding sigma
            if (it->second > this->sigma_sq)
                this->sigma_sq = it->second;
        }
    }

    this->sigma_sq /= 3;
    this->sigma_sq = 2 * (this->sigma_sq * this->sigma_sq);
    this->denominator = sqrt(2 * M_PI * this->sigma_sq);
}

// Initialize the graph nodos
template <class T>
void UnsupervisedOPF<T>::build_initialize()
{
    // Compute rho
    std::set<Pdist>::iterator it;
    for (size_t i = 0; i < this->nodes.size(); i++)
    {
        int n_neighbors = this->nodes[i].adj.size(); // A node may have more than k neighbors
        float div = this->denominator * n_neighbors;
        float sum = 0;

        for (it = this->nodes[i].adj.cbegin(); it != this->nodes[i].adj.cend(); ++it)
        {
            float dist = it->second;
            sum += expf((-dist * dist) / this->sigma_sq);
        }

        this->nodes[i].rho = sum / div;
    }

    // Compute delta
    this->delta = INF;
    for (size_t i = 0; i < this->nodes.size(); i++)
    {
        for (it = this->nodes[i].adj.begin(); it != this->nodes[i].adj.end(); ++it)
        {
            float diff = abs(this->nodes[i].rho - this->nodes[it->first].rho);
            if (diff != 0 && this->delta > diff)
                this->delta = diff;
        }
    }

    // And, finally, initialize each node
    this->queue.resize(this->nodes.size());
    for (size_t i = 0; i < this->nodes.size(); i++)
    {
        this->nodes[i].value = this->nodes[i].rho - this->delta;
        this->queue[i] = static_cast<int>(i);
    }
}

// Get the node with the biggest path value
// TODO: implement it in a more efficient way?
template <class T>
int UnsupervisedOPF<T>::get_max()
{
    float maxval = -INF;
    int maxidx = -1;
    int size = this->queue.size();
    for (int i = 0; i < size; i++)
    {
        int idx = this->queue[i];
        if (this->nodes[idx].value > maxval)
        {
            maxidx = i;
            maxval = this->nodes[idx].value;
        }
    }

    int best = this->queue[maxidx];
    int tmp = this->queue[size-1];
    this->queue[size-1] = this->queue[maxidx];
    this->queue[maxidx] = tmp;
    this->queue.pop_back();

    return best;
}

// OPF clustering
template <class T>
void UnsupervisedOPF<T>::cluster()
{
    // Cluster labels
    int l = 0;
    // Priority queue
    while (!this->queue.empty())
    {
        int s = this->get_max(); // Pop the highest value

        // If it has no predecessor, make it a prototype
        if (this->nodes[s].pred == -1)
        {
            this->nodes[s].label = l++;
            this->nodes[s].value = this->nodes[s].rho;
        }

        // Iterate and conquer over its neighbors
        for (auto it = this->nodes[s].adj.begin(); it != this->nodes[s].adj.end(); ++it)
        {
            int t = it->first;
            if (this->nodes[t].value < this->nodes[s].value)
            {
                float rho = std::min(this->nodes[s].value, this->nodes[t].rho);
                // std::cout << rho << " " << this->nodos[t].value << std::endl;
                if (rho > this->nodes[t].value)
                {
                    this->nodes[t].label = this->nodes[s].label;
                    this->nodes[t].pred = s;
                    this->nodes[t].value = rho;
                }
            }
        }
    }

    this->n_clusters = l;
}


// Fit the model
template <class T>
void UnsupervisedOPF<T>::fit(const Mat<T> &train_data)
{
    this->train_data = std::shared_ptr<const Mat<T>>(&train_data, [](const Mat<T> *p) {});
    this->nodes = std::vector<NodeKNN>(this->train_data->linhas);
    this->build_graph();
    this->build_initialize();
    if (!this->anomaly)
        this->cluster();
}

// Fit and prediz for all nodos
template <class T>
std::vector<int> UnsupervisedOPF<T>::fit_predict(const Mat<T> &train_data)
{
    this->fit(train_data);

    std::vector<int> labels(this->nodes.size());

    if (this->anomaly)
        for (size_t i = 0; i < this->nodes.size(); i++)
            labels[i] = (this->nodes[i].rho < this->thresh) ? 1 : 0;
    else
        for (size_t i = 0; i < this->nodes.size(); i++)
            labels[i] = this->nodes[i].label;

    return labels;
}

// Predict cluster pertinence
template <class T>
std::vector<int> UnsupervisedOPF<T>::predict(const Mat<T> &test_data)
{
    std::vector<int> preds(test_data.linhas);
    // For each test sample
    for (size_t i = 0; i < test_data.linhas; i++)
    {
        // Find the k nearest neighbors
        BestK bk(this->k);
        for (size_t j = 0; j < this->nodes.size(); j++)
        {
            if (i != j)
            {
                float dist;
                if (this->precomputed)
                    dist = test_data.em(i, j);
                else
                    dist = this->distance(test_data[i], this->train_data->linha(j), this->train_data->colunas);

                bk.insert(j, dist);
            }
        }

        // Compute the testing rho
        std::vector<Pdist> neighbors = bk.get_knn();
        int n_neighbors = neighbors.size();

        float div = this->denominator * n_neighbors;
        float sum = 0;

        for (int j = 0; j < n_neighbors; j++)
        {
            float dist = neighbors[j].second; // this->distances[i][*it]
            sum += expf((-dist * dist) / this->sigma_sq);
        }

        float rho = sum / div;

        if (this->anomaly)
        {
            // And returns anomaly detection based on graph density
            preds[i] = (rho < this->thresh) ? 1 : 0;
        }
        else
        {
            // And find which node conquers this test sample
            float maxval = -INF;
            int maxidx = -1;

            for (int j = 0; j < n_neighbors; j++)
            {
                int s = neighbors[j].first;  // idx, distancia
                float val = std::min(this->nodes[s].value, rho);
                if (val > maxval)
                {
                    maxval = val;
                    maxidx = s;
                }
            }

            preds[i] = this->nodes[maxidx].label;
        }
    }

    return preds;
}

// Quality metric
// From: A Robust Extension of the Mean Shift Algorithm using Optimum Path Forest
// Leonardo Rocha, Alexandre Falcao, and Luis Meloni
template <class T>
float UnsupervisedOPF<T>::quality_metric()
{
    if (this->anomaly)
        throw std::invalid_argument("Quality metric not implemented for anomaly detection yet");
    std::vector<float> w(this->n_clusters, 0);
    std::vector<float> w_(this->n_clusters, 0);
    for (size_t i = 0; i < this->train_data->linhas; i++)
    {
        int l = this->nodes[i].label;

        for (auto it = this->nodes[i].adj.begin(); it != this->nodes[i].adj.end(); ++it)
        {
            int l_ = this->nodes[it->first].label;
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

// Brute force method to find the best value of k
template <class T>
void UnsupervisedOPF<T>::find_best_k(Mat<float>& train_data, int kmin, int kmax, int step, bool precompute)
{
    std::cout << "precompute " << precompute << std::endl;
    float best_quality = INF;
    UnsupervisedOPF<float> best_opf;
    DistMat<float> distances;
    if (precompute)
        distances = DistMat<float>(train_data, this->distance);

    for (int k = kmin; k <= kmax; k += step)
    {
        // Instanciate and train the model
        UnsupervisedOPF<float> opf(k, false, 0, precompute, this->distance);
        if (precompute)
            opf.fit(distances);
        else
            opf.fit(train_data);

        std::cout << k << ": " << opf.n_clusters << std::endl;
        // Compare its clustering grade
        float quality = opf.quality_metric(); // Normalized cut
        if (quality < best_quality)
        {
            best_quality = quality;
            best_opf = opf;
        }

    }

    if (best_quality == INF)
    {
        std::ostringstream ss;
        ss << "No search with kmin " << kmin << ", kmax " << kmax << ", and step " << step << ". The arguments might be out of order.";
        std::cerr << ss.str() << std::endl;
        throw 0;
    }


    if (this->precomputed)
        this->train_data = std::shared_ptr<Mat<T>>(&distances, std::default_delete<Mat<T>>());
    else
        this->train_data = std::shared_ptr<Mat<T>>(&train_data, [](Mat<T> *p) {});

    this->k = best_opf.k;
    this->n_clusters = best_opf.n_clusters;
    this->nodes = best_opf.nodes;
    this->denominator = best_opf.denominator;
    this->sigma_sq = best_opf.sigma_sq;
    this->delta = best_opf.delta;
}

/*****************************************/
/*              Persistence              */
/*****************************************/

template <class T>
std::string UnsupervisedOPF<T>::serialize(uchar flags)
{
    if (this->precomputed)
        throw std::invalid_argument("Serialization for isPrecomputado OPF not implemented yet");

    // Open file
    std::ostringstream output   ;
    int n_samples = this->train_data->linhas;
    int n_features = this->train_data->colunas;

    // Output flags
    flags = 0; // For now, there are no user-defined flags
    if (this->anomaly)
        flags += SFlags::NaoSupervisionado_Anomalia;

    // Header
    escreveBinario<char>(output, "OPF", 3);
    escreveBinario<uchar>(output, Tipo::Agrupamento);
    escreveBinario<uchar>(output, flags);
    escreveBinario<uchar>(output, static_cast<uchar>(0)); // Reserved byte
    escreveBinario<int>(output, n_samples);
    escreveBinario<int>(output, n_features);

    // Scalar dado
    escreveBinario<int>(output, this->k);
    if (!this->anomaly)
        escreveBinario<int>(output, this->n_clusters);
    else
        escreveBinario<float>(output, this->thresh);
    escreveBinario<float>(output, this->denominator);
    escreveBinario<float>(output, this->sigma_sq);

    // Data
    for (int i = 0; i < n_samples; i++)
    {
        const T* data = this->train_data->linha(i);
        escreveBinario<T>(output, data, n_features);
    }

    // Nodes
    for (int i = 0; i < n_samples; i++)
    {
        escreveBinario<float>(output, this->nodes[i].value);
        if (!this->anomaly)
            escreveBinario<int>(output, this->nodes[i].label);
    }

    return output.str();
}

template <class T>
UnsupervisedOPF<T> UnsupervisedOPF<T>::unserialize(const std::string& contents)
{
    UnsupervisedOPF<float> opf;

    // Open stream
    std::istringstream ifs(contents); // , std::ios::in | std::ios::binary

    /// Header
    int n_samples;
    int n_features;
    char header[4];

    // Check if stream is an OPF serialization
    lerbinario<char>(ifs, header, 3);
    header[3] = '\0';
    if (strcmp(header, "OPF"))
        throw std::invalid_argument("Input is not an OPF serialization");

    // Get type and flags
    uchar type = lerBinario<uchar>(ifs);
    uchar flags = lerBinario<uchar>(ifs); // Flags byte
    lerBinario<uchar>(ifs); // reserved byte

    if (flags & SFlags::NaoSupervisionado_Anomalia)
        opf.anomaly = true;

    if (type != Tipo::Agrupamento)
        throw std::invalid_argument("Input is not an Unsupervised OPF serialization");

    // Data tamanho
    n_samples = lerBinario<int>(ifs);
    n_features = lerBinario<int>(ifs);

    // Scalar dado
    opf.k = lerBinario<int>(ifs);
    if (!opf.anomaly)
        opf.n_clusters = lerBinario<int>(ifs);
    else
    {
        opf.thresh = lerBinario<float>(ifs);
        opf.n_clusters = 2;
    }
    opf.denominator = lerBinario<float>(ifs);
    opf.sigma_sq = lerBinario<float>(ifs);

    /// Data
    // Temporary var to read dado, since opf's dadoDeTreinamento is const
    auto train_data = std::shared_ptr<Mat<T>>(new Mat<T>(n_samples, n_features), std::default_delete<Mat<T>>());
    // Read dado
    int size = n_samples * n_features;
    T* data = train_data->linha(0);
    lerbinario<T>(ifs, data, size);
    // Assign to opf
    opf.train_data = train_data;

    // Nodes
    opf.nodes = std::vector<NodeKNN>(n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        opf.nodes[i].value = lerBinario<float>(ifs);
        if (!opf.anomaly)
            opf.nodes[i].label = lerBinario<int>(ifs);
    }

    return std::move(opf);
}

/*****************************************/

}

#endif
