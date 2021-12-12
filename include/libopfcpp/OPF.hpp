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
#include <iostream>

namespace opf {

#define INF std::numeric_limits<float>::infinity()
#define NIL (-1)

// Função genérica de distância
    template<class T>
    using funcaoDistancia = std::function<T(const T *, const T *, size_t)>;

/*****************************************/
/************** Tipo Matriz **************/
/*****************************************/
/**
 * Classe n-dimensional de matriz densa
 * @tparam T
 */
    template<class T=float>
    class Mat {
    protected:
        std::shared_ptr<T> dado;
    public:
        size_t linhas{}, colunas{};
        size_t tamanho{};
        size_t passo{};

        // Construtores
        Mat();

        Mat(size_t linhas, size_t colunas);

        virtual T *linha(size_t i);

        virtual T *operator[](size_t i);

        const virtual T *operator[](size_t i) const;

        Mat<T> &operator=(const Mat<T> &outro);
    };

    template<class T>
    Mat<T>::Mat() {
        this->linhas = this->colunas = this->tamanho = this->passo = 0;
    }

    template<class T>
    Mat<T>::Mat(size_t linhas, size_t colunas) {
        this->linhas = linhas;
        this->colunas = colunas;
        this->tamanho = linhas * colunas;
        this->passo = colunas;
        this->dado = std::shared_ptr<T>(new T[this->tamanho], std::default_delete<T[]>());
    }

    template<class T>
    T *Mat<T>::linha(size_t i) {
        size_t indice = i * this->passo;
        return this->dado.get() + indice;
    }

    template<class T>
    T *Mat<T>::operator[](size_t i) {
        size_t indice = i * this->passo;
        return this->dado.get() + indice;
    }

    template<class T>
    const T *Mat<T>::operator[](size_t i) const {
        size_t indice = i * this->passo;
        return this->dado.get() + indice;
    }

    template<class T>
    Mat<T> &Mat<T>::operator=(const Mat<T> &outro) {
        if (this != &outro) {
            this->linhas = outro.linhas;
            this->colunas = outro.colunas;
            this->tamanho = outro.tamanho;
            this->dado = outro.dado;
            this->passo = outro.passo;
        }

        return *this;
    }

    /*****************************************/

// Função de distância padrão
    template<class T>
    T distanciaEuclidiana(const T *a, const T *b, size_t tamanho) {
        T soma = 0;
        for (size_t i = 0; i < tamanho; i++) {
            soma += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return (T) sqrt(soma);
    }


/*********************************************/
/************ Estruturas de Dados ************/
/*********************************************/

/**
 * Códigos de cores para o algoritmo de Prim
 */
    enum Color {
        BRANCO, // Novo vértice
        CINZA,  // Na Heap
        PRETO  // Já visto
    };

/**
 * Classe simples para armazenar informações de vértice
 */
    class Vertice {
    public:
        Vertice() {
            this->cor = BRANCO;
            this->predecessor = -1;
            this->custo = INF;
            this->isPrototipo = false;
        }

        size_t indice{};      // Índice na lista - facilita as pesquisas
        Color cor;            // Cor na Heap. branco: nunca visitada, cinza: na Heap, preto: removido da Heap
        float custo;          // Custo para alcançar o vértice
        int rotuloVerdade{};  // Valor de referência
        int rotulo{};         // Rótulo atribuído
        int predecessor;      // Vértice antecessor
        bool isPrototipo;     // Se o vértice é um protótipo
    };

/**
 * Estrutura de dados de Heap para usar como uma fila de prioridade
 *
 */
    class Heap {
    private:
        std::vector<Vertice> *vertices; // Uma referência para o vetor de contêiner original
        std::vector<Vertice *> vec;   // Um vetor de ponteiros para construir a Heap em cima

        static bool comparaElemento(const Vertice *estruturaHeapEsquerda, const Vertice *estruturaHeapDireita) {
            return estruturaHeapEsquerda->custo >= estruturaHeapDireita->custo;
        }

    public:
        // Construtor de tamanho
        Heap(std::vector<Vertice> *vertices, const std::vector<int> &rotulos) {
            this->vertices = vertices;
            size_t n = vertices->size();
            this->vec.reserve(n);
            for (size_t i = 0; i < n; i++) {
                (*this->vertices)[i].indice = i;
                (*this->vertices)[i].rotuloVerdade = (*this->vertices)[i].rotulo = rotulos[i];
            }
        }

        /**
         * Insere o novo elemento na Heap
         * @param item elemento
         * @param custo custo
         */
        void push(int item, float custo) {
            // Atualiza o valor de custo do vértice
            (*this->vertices)[item].custo = custo;

            // Já presente na Heap
            if ((*this->vertices)[item].cor == CINZA)
                make_heap(this->vec.begin(), this->vec.end(), comparaElemento); // Remake the Heap

                // Novo na Heap
            else if ((*this->vertices)[item].cor == BRANCO) {
                (*this->vertices)[item].cor = CINZA;
                this->vec.push_back(&(*this->vertices)[item]);
                push_heap(this->vec.begin(), this->vec.end(), comparaElemento); // Push new item to the Heap
            }
            // Note that black items can not be inserted into the Heap
        }

        /**
         * Atualiza o custo do item sem atualizar a Heap
         * @param item item
         * @param custo custo
         */
        void atualizaCusto(int item, float custo) {
            // Atualiza o valor de custo do vértice
            (*this->vertices)[item].custo = custo;
            if ((*this->vertices)[item].cor == BRANCO) {
                (*this->vertices)[item].cor = CINZA;
                this->vec.push_back(&(*this->vertices)[item]);
            }
        }

        /**
         * Atualiza a Heap.
         * Isso é usado após várias chamadas para atualizaCusto, a fim de reduzir o número de chamadas para make_heap.
         */
        void heapify() {
            make_heap(this->vec.begin(), this->vec.end(), comparaElemento); // Remake the Heap
        }

        /**
         * Remova e devolva o primeiro elemento da Heap
         * @return primeiro elemento da Heap
         */
        int pop() {
            // Obtém e marca o primeiro elemento
            Vertice *frente = this->vec.front();
            frente->cor = PRETO;
            // O remove da Heap
            pop_heap(this->vec.begin(), this->vec.end(), comparaElemento);
            this->vec.pop_back();
            // E o retorna
            return frente->indice;
        }

        /**
         * Verifica se a Heap está vazia
         * @return true se estive vazia
         */
        bool isVazia() {
            return this->vec.empty();
        }

    };

/*****************************************/



/*****************************************/
/****************** OPF ******************/
/*****************************************/

/******** Supervisionado ********/
    template<class T=float>
    class OPFSupervisionado {
    private:
        // Modelo
        Mat<T> dadoDeTreinamento; // Dados de treinamento (vetores originais ou matriz de distância)
        std::vector<Vertice> vertices;  // Modelo aprendido

        // Lista de vértices ordenados pelo custo. Útil para acelerar a classificação
        // Não é definido com size_t para reduzir o uso de memória, visto que o ML pode lidar com dados grandes
        std::vector<unsigned int> nodosOrdenados;

        // Opções
        bool isPrecomputado{};
        funcaoDistancia<T> distancia;

        void prototipoDePrim(const std::vector<int> &rotulos);


    public:
        explicit OPFSupervisionado(bool isPrecomputado = false, funcaoDistancia<T> distancia = distanciaEuclidiana<T>);

        void ajusta(const Mat<T> &treinamento, const std::vector<int> &rotulos);

        std::vector<int> prediz(const Mat<T> &dadoDeTeste);

    };

    template<class T>
    OPFSupervisionado<T>::OPFSupervisionado(bool isPrecomputado, funcaoDistancia<T> distancia) {
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
    template<class T>
    void OPFSupervisionado<T>::prototipoDePrim(const std::vector<int> &rotulos) {
        this->vertices = std::vector<Vertice>(this->dadoDeTreinamento.linhas);
        Heap cabeca(&this->vertices, rotulos); // Heap como uma fila de prioridades

        // Primeiro vértice arbitrário
        cabeca.push(0, 0);

        while (!cabeca.isVazia()) {
            // Pega a cabeça da Heap e marca-a preta
            size_t s = cabeca.pop();

            // Definição de protótipo
            int pred = this->vertices[s].predecessor;
            if (pred != NIL) {
                // Encontra pontos na fronteira entre duas classes...
                if (this->vertices[s].rotuloVerdade != this->vertices[pred].rotuloVerdade) {
                    // E os define como protótipos
                    this->vertices[s].isPrototipo = true;
                    this->vertices[pred].isPrototipo = true;
                }
            }

            // Seleção de arestas
#pragma omp parallel for default(shared)
            for (size_t t = 0; t < this->vertices.size(); t++) {
                // Se os vértices diferem e t não foi retirado da Heap (marcado preto)
                if (s != t && this->vertices[t].cor != PRETO) {
                    // Calcula o peso
                    float peso;
                    if (this->isPrecomputado)
                        peso = this->dadoDeTreinamento[s][t];
                    else
                        peso = this->distancia(this->dadoDeTreinamento[s], this->dadoDeTreinamento[t],
                                               this->dadoDeTreinamento.colunas);

                    // Atribui se menor do que o valor atual
                    if (peso < this->vertices[t].custo) {
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
 * Treina o modelo com o dado e rótulos.
 * @tparam T
 * @param treinamento
 *          - vetores de recurso original [n_samples, n_features] -- se isPrecomputado == false
 *          - matriz de distancia [n_samples, n_samples] -- se isPrecomputado == true
 * @param rotulos
 *          - valores dos rotulo-verdade [n_samples]
 */
    template<class T>
    void OPFSupervisionado<T>::ajusta(const Mat<T> &treinamento, const std::vector<int> &rotulos) {
        if ((size_t) treinamento.linhas != rotulos.size())
            throw std::invalid_argument("[OPF/ajusta] Erro: tamanho dos dados não correspondem ao tamanho dos rótulos"
                                        + std::to_string(treinamento.linhas) + " x " + std::to_string(rotulos.size()));

        // Armazenar referência de dados para testes
        this->dadoDeTreinamento = treinamento;

        // Modelo de inicialização
        this->prototipoDePrim(rotulos); // Encontra protótipos
        Heap cabeca(&this->vertices, rotulos); // Heap como uma fila de prioridade

        // Inicialização
        for (Vertice &vertice: this->vertices) {
            vertice.cor = BRANCO;
            // Protótipos de custo 0, não ter antecessor e povoar o Heap
            if (vertice.isPrototipo) {
                vertice.predecessor = NIL;
                vertice.custo = 0;
            } else // Outros vértices iniciam com custo = INF
            {
                vertice.custo = INF;
            }
            // Visto que todos os vértices estão conectados a todos os outros
            cabeca.push(vertice.indice, vertice.custo);
        }

        // Lista de vértices ordenados por custo
        // Útil para acelerar a classificação
        this->nodosOrdenados.reserve(this->vertices.size());

        // consome a fila
        while (!cabeca.isVazia()) {
            int s = cabeca.pop();
            this->nodosOrdenados.push_back(s);

            // Itera sobre todos os vizinhos
#pragma omp parallel for default(shared)
            for (int t = 0; t < (int) this->vertices.size(); t++) {
                if (s != t && this->vertices[s].custo < this->vertices[t].custo) {
                    // Computa o peso
                    float peso;
                    if (isPrecomputado)
                        peso = this->dadoDeTreinamento[s][t];
                    else
                        peso = distancia(this->dadoDeTreinamento[s], this->dadoDeTreinamento[t],
                                         this->dadoDeTreinamento.colunas);

                    float cost = std::max(peso, this->vertices[s].custo);
                    if (cost < this->vertices[t].custo) {
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
 * Classifica um conjunto de amostras usando um modelo treinado por OPFSupervisionado::ajusta.
 * @tparam T
 * @param dadoDeTeste
 *          - vetores de recurso original [n_samples, n_features] -- se isPrecomputado == false
 *          - matriz de distancia [n_samples, n_samples] -- se isPrecomputado == true
 * @return
 * - previsões:
 * - vetor<int> de tamanho [n_test_samples] com saídas de classificação.
 */
    template<class T>
    std::vector<int> OPFSupervisionado<T>::prediz(const Mat<T> &dadoDeTeste) {
        int n_amostrasDeTeste = (int) dadoDeTeste.linhas;
        int n_amostrasDeTreinamento = (int) this->vertices.size();

        // Previsões de saída
        std::vector<int> previsoes(n_amostrasDeTeste);

#pragma omp parallel for default(shared)
        for (int i = 0; i < n_amostrasDeTeste; i++) {
            int indice = this->nodosOrdenados[0];
            int diceMinimo = 0;
            T custoMinimo = INF;
            T peso;

            // 'nodosOrdenados' contém índices amostrais ordenados pelo custo, portanto,
            // se a melhor conexão atual custa menos do que o próximo vértice, é inútil continuar procurando.
            for (int j = 0; j < n_amostrasDeTreinamento && custoMinimo > this->vertices[indice].custo; j++) {
                // Obtém o próximo vértice na lista ordenada
                indice = this->nodosOrdenados[j];

                // Calcular sua distância até o ponto de consulta
                if (isPrecomputado)
                    peso = dadoDeTeste[i][indice];
                else
                    peso = distancia(dadoDeTeste[i], this->dadoDeTreinamento[indice], this->dadoDeTreinamento.colunas);

                // O custo corresponde ao máximo entre a distância e o custo de referência
                float custo = std::max(peso, this->vertices[indice].custo);

                if (custo < custoMinimo) {
                    custoMinimo = custo;
                    diceMinimo = indice;
                }
            }

            previsoes[i] = this->vertices[diceMinimo].rotulo;
        }

        return previsoes;
    }

/*****************************************/

/*****************************************/
/**********   Acesso de dados   **********/
}
#endif