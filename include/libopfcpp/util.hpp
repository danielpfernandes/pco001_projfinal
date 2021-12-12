/******************************************************
 * Utilitários para OPF.hpp                           *
 *                                                    *
 * Projeto original: Thierry Moreira, 2019            *
 *                                                    *
 * Adaptado por: Daniel P Fernandes, Natalia S        *
 * Sanchez & Alexandre L Sousa                        *
 *                                                    *
 ******************************************************/

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



#ifndef UTIL_HPP
#define UTIL_HPP

#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <cmath>

#include "libopfcpp/OPF.hpp"

using namespace std::chrono;

namespace opf {

/**
 * Imprime um dado vetor
 * @tparam T template
 * @param v vetor
 * @param tamanho tamanho do vetor
 */
    template<class T>
    void imprimeVetor(T *v, int tamanho) {
        std::cout << "[";
        for (int i = 0; i < tamanho; i++)
            std::cout << v[i] << ' ';
        std::cout << "]";
    }

/**
 * Imprime uma dada matriz
 * @tparam T template
 * @param m matriz
 */
    template<class T>
    void imprimeMatriz(Mat <T> m) {
        for (int i = 0; i < m.linhas; i++) {
            T *row = m.linha(i);
            imprimeVetor(row, m.colunas);
            std::cout << '\n';
        }
        std::cout << std::endl;
    }

/**
 * Lê o rotulo das matrizes
 * @tparam T template
 * @param nomeDoArquivo Nome do arquivo
 * @param dado Dado da matriz
 * @param rotulos Rótulos do conjunto de dados
 * @return true e os rótulos foram lidos com sucesso
 */
    template<class T>
    bool lerRotuloDasMatrizes(const std::string &nomeDoArquivo, Mat <T> &dado, std::vector<int> &rotulos) {
        std::ifstream arquivo(nomeDoArquivo, std::ios::in | std::ios::binary);

        if (!arquivo.is_open()) {
            std::cerr << "[util/lerRotuloDasMatrizes] Could not open arquivo: " << nomeDoArquivo << std::endl;
            return false;
        }

        int linhas, colunas;
        arquivo.read((char *) &linhas, sizeof(int));
        arquivo.read((char *) &colunas, sizeof(int));

        dado = Mat<T>(linhas, colunas);
        rotulos = std::vector<int>(linhas);

        int rotulo;
        T val;
        for (int i = 0; i < linhas; i++) {
            // rotulo
            arquivo.read((char *) &rotulo, sizeof(int));
            rotulos[i] = rotulo;

            for (int j = 0; j < colunas; j++) {
                arquivo.read((char *) &val, sizeof(T));
                dado[i][j] = val;
            }
        }
        arquivo.close();

        return true;
    }

    /**
     * Converter o vetor de um dataset para o formato string
     * @tparam T
     * @param v Vetor
     * @param tamanho tamanho do vetor
     * @return String do vetor convertido
     */
    template<class T>
    std::string escreveVetor(T *v, int tamanho) {
        std::string vetor;
        vetor.append("[");
        for (int i = 0; i < tamanho; i++)
            vetor.append(std::to_string(v[i]));
        vetor.append(" ");
        vetor.append("]");
        return vetor;
    }

    /**
     * Converte uma matriz de dataset para o formato string
     * @tparam T
     * @param m Matriz
     * @return vetor de strings da matriz convertida
     */
    template<class T>
    std::vector<std::string> stringMatriz(Mat <T> m) {
        std::vector<std::string> matriz;
        for (int i = 0; i < m.linhas; i++) {
            T *row = m.linha(i);
            matriz.push_back(escreveVetor(row, m.colunas));
        }
        return matriz;
    }

    /**
     * Cria um arquivo com os dados de uma dada matriz (teste/treinamento/etc)
     * @tparam T
     * @param nomeArquivo Nome do arquivo
     * @param tipo sufixo do arquivo
     * @param matriz matriz
     */
    template<class T>
    void armazenaDados(std::string nomeArquivo, std::string tipo, Mat <T> matriz) {
        FILE *arquivo;

        arquivo = fopen(nomeArquivo.append("." + tipo).c_str(), "a");
        std::vector<std::string> matrizDeTreinamento = stringMatriz(matriz);

        system_clock::time_point p = system_clock::now();
        std::time_t t = system_clock::to_time_t(p);
        fprintf(arquivo, "%s", "\n================================================\n");
        fprintf(arquivo, "%s", std::ctime(&t));
        fprintf(arquivo, "%s", "================================================\n");

        for (std::string entrada: matrizDeTreinamento) {
            fprintf(arquivo, "%s\n", entrada.c_str());
        }
        fclose(arquivo);
    }

    /**
     * Cria um arquivo com os dados de uma dado vetor (por exemplo, valores de referência - ground truth)
     * @tparam T
     * @param nomeArquivo Nome do arquivo
     * @param vetor vetor
     */
    template<class T>
    void armazenaDados(std::string nomeArquivo, std::vector<T> vetor) {

        FILE *arquivo;
        arquivo = fopen(nomeArquivo.append(".referencia").c_str(), "a");

        system_clock::time_point p = system_clock::now();
        std::time_t t = system_clock::to_time_t(p);
        fprintf(arquivo, "%s", "\n================================================\n");
        fprintf(arquivo, "%s", std::ctime(&t));
        fprintf(arquivo, "%s", "================================================\n");

        for (T value: vetor) {
            fprintf(arquivo, "%d\n", value);
        }
        fclose(arquivo);
    }

/**
 * Fornece índices de treino/teste para dividir dados em conjuntos de treinos/testes. Este objeto de validação cruzada
 * é uma mesclagem de StratifiedKFold e ShuffleSplit, que retorna dobras aleatórias estratificadas.
 * As dobras são feitas preservando a porcentagem de amostras para cada classe.
 */
    class StratifiedShuffleSplit {
    private:
        float razaoDeTreinamento;
        std::default_random_engine random_engine;

    public:
        StratifiedShuffleSplit(float razaoDeTreinamento = 0.5) : razaoDeTreinamento(razaoDeTreinamento) {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            this->random_engine = std::default_random_engine(seed);
        }

        std::pair<std::vector<int>, std::vector<int>> split(const std::vector<int> &rotulos);
    };

// índices de treino, índices de testes
    std::pair<std::vector<int>, std::vector<int>> StratifiedShuffleSplit::split(const std::vector<int> &rotulos) {
        std::map<int, int> totais, destino, atual;
        std::map<int, int>::iterator it;
        std::pair<std::vector<int>, std::vector<int>> divisoes;

        int tamanhoTeste, tamanhoTreino = 0;

        for (int r: rotulos)
            totais[r]++;

        // Encontra o número de amostras para cada classe
        for (it = totais.begin(); it != totais.end(); ++it) {
            destino[it->first] = (int) round((float) it->second * this->razaoDeTreinamento);
            tamanhoTreino += destino[it->first];
        }
        tamanhoTeste = rotulos.size() - tamanhoTreino;

        // Inicializa a saída
        divisoes.first.resize(tamanhoTreino);
        divisoes.second.resize(tamanhoTeste);

        // Embaralha os índices
        std::vector<int> idx(rotulos.size());
        for (unsigned int i = 0; i < rotulos.size(); i++)
            idx[i] = i;

        std::shuffle(idx.begin(), idx.end(), this->random_engine);

        // Atribui dobras
        int j, l;
        int indiceDeTreino = 0, indiceDeTeste = 0;
        for (unsigned int i = 0; i < rotulos.size(); i++) {
            j = idx[i];
            l = rotulos[j];

            if (atual[l] < destino[rotulos[j]]) {
                divisoes.first[indiceDeTreino++] = j;
                atual[l]++;
            } else {
                divisoes.second[indiceDeTeste++] = j;
            }
        }

        return divisoes;
    }

/**
 * Indexa os dados de acordo com um vetor de índices num vetor de saída
 * @tparam T
 * @param dado Conjunto de dados (dados, rótulos de dados etc)
 * @param indices Ìndices obtidos de um processo de StratifiedShuffleSplit
 * @param saida Vetor de dados que armazenará o resultado desta função
 */
    template<class T>
    void indicePorLista(const std::vector<T> &dado, const std::vector<int> &indices, std::vector<T> &saida) {
        int tamanho = indices.size();
        saida = std::vector<T>(tamanho);

        for (int i = 0; i < tamanho; i++) {
            saida[i] = dado[indices[i]];
        }
    }

/**
 * Indexa os dados de acordo com um vetor de índices numa matriz de saída
 * @tparam T
 * @param dado Conjunto de dados (dados de treinamento etc)
 * @param indices Ìndices obtidos de um processo de StratifiedShuffleSplit
 * @param saida Matriz de dados que armazenará o resultado desta função
 */
    template<class T>
    void indicePorLista(const Mat <T> &dado, const std::vector<int> &indices, Mat <T> &saida) {
        int size = (int) indices.size();
        saida = Mat<T>(size, dado.colunas);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < dado.colunas; j++)
                saida[i][j] = dado[indices[i]][j];
        }
    }

/**
 * Calcula a acurácia baseado no valor de referência e nas previsões dos dados de teste
 * @param valorDeReferencia valor de referência
 * @param previsoes previsões dos dados de teste
 * @return o valor da acurácia
 */
    float acuracia(const std::vector<int> &valorDeReferencia, const std::vector<int> &previsoes) {
        if (valorDeReferencia.size() != previsoes.size()) {
            std::cerr << "[util/acuracia] Error: o valor de referência e os tamanhos de previsão não correspondem.. "
                      << valorDeReferencia.size() << " x " << previsoes.size() << std::endl;
        }

        auto n = static_cast<float>(valorDeReferencia.size());
        float acuracia = 0;
        for (int i = 0; i < n; i++)
            if (valorDeReferencia[i] == previsoes[i])
                acuracia++;

        return acuracia / n;
    }
}

#endif
