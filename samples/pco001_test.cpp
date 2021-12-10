/******************************************************
 * Exemplo de uso do OPFSupervisionado em 3 datasets como *
 * requisito parcial do projeto final de PCO001 da    *
 * Universidade Federal de Itajubá - MG               *
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



#include <vector>
#include <iostream>
#include <string>

#include <sys/time.h>
#include <ctime>
#include <cstdio>
#include <cassert>


#include "libopfcpp/OPF.hpp"
#include "libopfcpp/util.hpp"

using namespace std;
using namespace opf;


typedef timeval timer;
#define outchannel stdout
#define TIMING_START() timer TM_start, TM_now, TM_now1;\
    gettimeofday(&TM_start,NULL);\
    TM_now = TM_start;
#define SECTION_START(M) gettimeofday(&TM_now,NULL);\
    fprintf(outchannel,"================================================\nIniciando a medição de %s\n",M);
#define TIMING_SECTION(M, medida) gettimeofday(&TM_now1,NULL);\
    *(medida)=(TM_now1.tv_sec-TM_now.tv_sec)*1000.0 + (TM_now1.tv_usec-TM_now.tv_usec)*0.001;\
    fprintf(outchannel,"%.3fms:\tSEÇÃO %s\n",*(medida),M);\
    TM_now=TM_now1;
#define TIMING_END() gettimeofday(&TM_now1,NULL);\
    fprintf(outchannel,"\nTotal time: %.3fs\n================================================\n",\
      	 (TM_now1.tv_sec-TM_start.tv_sec) + (TM_now1.tv_usec-TM_start.tv_usec)*0.000001);


/**
 *  Este exemplo treina e testa o modelo em três datasets:
 *      - Iris Data Set (UCI Machine Learning Repository)
 *      - Banana Dataset
 *      - Diabetic Retinopathy Debrecen Data Set (Balint Antal, Andras Hajdu: An ensemble-based system for automatic
 *        screening of diabetic retinopathy, Knowledge-Based Systems 60 (April 2014), 20-27.)
 *
 *  Para cada conjunto de dados, são calculadas a precisão de teste e o tempo de execução para o uso regular e
 *  usando matrizes de distância pré-computadorizado.
 */
int main(int argc, char *argv[])
{
    vector<string> datasets = {"data/pco001/iris.dat", "data/pco001/banana.dat", "data/pco001/messidor_features.dat"};
    TIMING_START();

    vector<vector<float>> times(5, vector<float>(datasets.size()));
    float medida;

    for (unsigned int i = 0; i < datasets.size(); i++)
    {
        string dataset = datasets[i];
        Mat<float> dado;
        vector<int> rotulos;

        // Ler dados
        lerRotuloDasMatrizes(dataset, dado, rotulos);

        // Divide
        SECTION_START(dataset.c_str());
        printf("Tamanho dos dados: Instâncias (%lu) x Atributos (%lu)\n\n", dado.linhas, dado.colunas);

        printf("Preparando dado\n");
        StratifiedShuffleSplit sss(0.5);
        pair<vector<int>, vector<int>> divisoes = sss.split(rotulos);

        TIMING_SECTION("Divisão dos dados", &medida);

        Mat<float> dadoDeTreinamento, dadoDeTeste;
        vector<int> rotulosDeTreinamento, valorDeReferencia;

        indicePorLista<float>(dado, divisoes.first, dadoDeTreinamento);
        indicePorLista<float>(dado, divisoes.second, dadoDeTeste);

        indicePorLista<int>(rotulos, divisoes.first, rotulosDeTreinamento);
        indicePorLista<int>(rotulos, divisoes.second, valorDeReferencia);

        TIMING_SECTION("indexando", &medida);

        // *********** Tempo de treinamento ***********
        printf("\nExecutando OPF...\n");

        // Classificador de treinamento
        OPFSupervisionado<float> opf;
        opf.ajusta(dadoDeTreinamento, rotulosDeTreinamento);

        TIMING_SECTION("Treinamento de OPF", &medida);
        times[0][i] = medida;
        
        // E previsão dos dados de teste
        vector<int> previsoes = opf.prediz(dadoDeTeste);

        TIMING_SECTION("Testando OPF", &medida);
        times[1][i] = medida;
        
        // Medindo acurácia
        float acc = acuracia(valorDeReferencia, previsoes);
        printf("Acurácia: %.3f%%\n", acc * 100);

        // Imprimindo dados
        printf("Matriz de treinamento: ");
        imprimeMatriz(dadoDeTreinamento);

        printf("Matriz de teste: ");
        imprimeMatriz(dadoDeTeste);

        printf("Dados de referência: ");
        //imprimeVetor(&valorDeReferencia, 100);
    }

    FILE *arquivo;
    arquivo = fopen("timing.txt", "a");
    for (size_t i = 0; i < datasets.size(); i++)
        fprintf(arquivo, "%s;%.3f;%.3f;%.3f;%.3f;%.3f\n", datasets[i].c_str(), times[0][i], times[1][i], times[2][i], times[3][i], times[4][i]);
    fclose(arquivo);

    arquivo = fopen("training.txt", "a");
    for (size_t i = 0; i < datasets.size(); i++)
        fprintf(arquivo, "%s;%.3f\n", datasets[i].c_str(), times[0][i]);
    fclose(arquivo);

    TIMING_END();

    return 0;
}

