<div style="display: flex; align-items: center;">
  <img src="TUCA_logo.jpg" alt="Grupo de Pesquisa em Mecânica dos Fluidos Computacional - ICMC - USP" width="100" height="100" style="margin-right: 20px;">
  <h1>Conversão de Dados Anemométricos com Rede Neural MLP</h1>
</div>

## Introdução

Este projeto, foi desenvolvido como um projeto de Iniciação Científica na Universidade de São Paulo, campus São Carlos, estando submetido ao __Grupo de Pesquisa em Mecânica dos Fluidos Computacional__*__ no ICMC - USP. A pesquisa propõe uma abordagem inovadora para a conversão de dados de tensão coletados por sensores hot-film a 2kHz em dados de velocidade tridimensional. Utilizando uma rede neural ___MultiLayer Perceptron___ (MLP), conseguiu-se traduzir esses dados sem a necessidade de utilizar a _Lei de King_ ou outros métodos analíticos tradicionais.

## Objetivo

O objetivo principal deste projeto é desenvolver uma ferramenta (algoritmo) capaz de converter dados de tensão de um sensor hot-film em dados de velocidade tridimensional (eixos x, y, z), com o auxílio de dados de velocidade coletados por um sensor sônico a 20Hz. A rede neural MLP foi escolhida principalmente pela sua simplicidade e adequação à demanda.

## Motivação

A _Lei de King_, tradicionalmente utilizada para converter tensão em velocidade, apresenta variações nos seus coeficientes ao longo do dia, influenciadas por fatores como temperatura, umidade e intensidade do ar, por exemplo. Essas diversas variações comprometem a precisão da conversão. Com o uso de uma rede neural MLP, buscou-se superar essas limitações, oferecendo uma solução mais robusta e precisa, que se adapta automaticamente às variações ambientais.

## Metodologia

### Estrutura da Rede Neural

- **Entradas**: 3 tensões (eixos x, y, z)
- **Saídas**: 3 velocidades (eixos x, y, z)
- **Arquitetura**: MultiLayer Perceptron (MLP) com múltiplas camadas ocultas e neurônios por camada ajustáveis.

### Processamento de Dados

1. **Coleta de Dados**: Sensores hot-film a 2kHz e sônicos a 20Hz.
2. **Preparação dos Dados**: Organização e limpeza dos dados para treinamento e teste.
3. **Treinamento**: Uso de dados sintéticos e reais para treinar a rede, com validação cruzada para evitar overfitting.
4. **Avaliação**: Análise dos resultados através de métricas como RMSE e QPE, e visualização dos periodogramas para validar a manutenção das propriedades turbulentas dos dados.

## Resultados

Os testes realizados com diferentes conjuntos de dados (puros, com ruído e com ângulo relativo entre os sensores) demonstraram a eficácia da rede neural MLP na conversão de dados de tensão em velocidade. Os principais achados incluem:

- **Dados Sintéticos Puros**: Erro médio quadrático absoluto (RMSE) de 0,065 m/s, com erro relativo entre 0,282% e 0,381%.
- **Dados Sintéticos com Ruído**: Desempenho robusto com erro menor que 4%, superando métodos analíticos tradicionais.
- **Dados Reais com Ângulo de Ataque Diferente**: Correção eficaz de erros geométricos, simplificando a instalação dos sensores.

## Conclusões

Este estudo reforça a viabilidade e a eficácia do uso de redes neurais para a conversão de dados anemométricos. A abordagem proposta não só simplifica o processo de coleta e análise de dados de sensores hot-film, mas também oferece uma solução mais precisa e eficiente em comparação aos métodos analíticos tradicionais.

## Perspectivas Futuras

- Exploração de arquiteturas de redes neurais mais complexas, como LSTM ou redes recorrentes, para capturar padrões temporais mais elaborados.
- Coleta e treinamento da rede com um volume maior de dados reais em diferentes condições ambientais para aprimorar ainda mais a precisão e robustez do modelo.
- Aplicação da metodologia em outras áreas de sensoriamento além da anemometria.

## Contribuições

Este projeto é um exemplo do potencial das redes neurais em substituir métodos analíticos tradicionais em aplicações de sensoriamento. Desenvolvido no contexto acadêmico, o código e a documentação estão disponíveis open source no GitHub, promovendo a ciência aberta e a colaboração entre pesquisadores.

---

Desenvolvido no âmbito de uma Iniciação Científica no ICMC - USP, São Carlos, com apoio do Grupo de Pesquisa em Mecânica dos Fluidos Computacional.
