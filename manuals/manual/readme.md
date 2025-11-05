# Manual de Uso

## Antes de Tudo

1. Após ter instalado o virtual environment com os passos do arquivo `installation.md`, vá para a pasta do projeto (será a mesma em que está este manual, `./hotfilm_NN`) e passe a usá-la para todos os procedimentos;

2. Execute o comando para habilitar o ambiente virtual:
    ```bash
    $ source hotfilm_env/bin/activate
    ```
    **Obs:** Caso queira sair do ambiente, execute:
    ```bash
    $ deactivate
    ```

3. Todos os comandos e interações com a rede são executadas com base em `./hotfilm_NN` localmente.

O identificador `"SERIE"` é carregado desde os primeiros arquivos e comandos até o final dos procedimentos, escolha-o estrategicamente para uma boa organização de seus projetos. Nos exemplos a seguir o identificador de série escolhido como exemplo é `"xxx"`.

## Treinamento

### Treinar uma Rede

Para treinar um modelo, precisa-se de:
- Um conjunto de dados de tensão por tempo a 2000Hz do sensor hotfilm;
- Um conjunto de dados de velocidade por tempo a 20Hz do sensor sônico.

1. **Preparação dos dados para treino**:
    Após a coleta dos dados em `.csv`, crie uma pasta única no formato `collected_data_xxx` dentro do diretório:
    ```
    data/raw_data/raw_train
    ```

2. **Organização dos arquivos**:
    Depois de criada, a pasta deve conter dois arquivos: `hotfilm_xxx.csv` e `sonic_xxx` da seguinte forma:
    ```
    data/raw_data/raw_train/collected_data_xxx/hotfilm_xxx.csv
    data/raw_data/raw_train/collected_data_xxx/sonic_xxx.csv
    ```
    **Exemplos**:
    ```
    collected_data_xxx
    ├── hotfilm_xxx.csv
    └── sonic_xxx.csv
    ```

3. **Criação do dataframe final**:
    Com os dados dentro da pasta, cria-se o dataframe final que será usado para o treino, para isso, execute:
    ```bash
    $ python3 csv_maker.py train {serie}
    ```
    **Exemplo**:
    ```bash
    $ python3 csv_maker.py train xxx
    ```

4. **Início do treinamento**:
    Com o dataset de treinamento pronto e dentro da pasta:
    ```
    data/train/
    ```
    Pode-se começar o treinamento, com o seguinte comando:
    ```bash
    $ python3 train_mlp.py {serie}
    ```
    **Exemplo**:
    ```bash
    $ python3 train_mlp.py xxx
    ```
    O treinamento pode levar de alguns minutos a algumas horas, dependendo da especificação da máquina local que processará os cálculos matriciais. Ao concluir o treinamento com suficientes 1000 epochs (parametrizável em `train_mlp.py`), serão apresentadas imagens referentes ao treinamento para conferência do usuário. Certifique-se. Para finalizar, basta fechar as imagens e o algoritmo irá finalizar.

5. **Resultados**:
    Os metadados serão salvos na pasta:
    ```
    data/train/train_results/
    ```
    Neste diretório acima será inserida uma planilha com as informações do treinamento e o nome do modelo respectivo a esse treinamento.
    O modelo será salvo na pasta `modelos/` como:
    ```
    modelos/model_mlp_xxx.pth
    ```

### Observações de Treinamento

Dentro do arquivo `train_mlp.py` há hiperparâmetros que podem ser alterados conforme a necessidade do treinamento:
- `EPOCHS`: Determina a quantidade de vezes que o dataset passará pela rede de treinamento (1000 - padrão);
- `EXPORT_DATA`: Exporta arquivos `.csv` (metadados) para analisar o resultado da rede;
- `GRAPHS`: Mostrar os gráficos ou não;
- `SAVE`: Salvar o modelo ou não;
- `GPU`: 0 para uso da CPU (RECOMENDADO) | 1 para treino em GPU (código não otimizado para GPU, NÃO RECOMENDADO);
- `LOCAL`: 0 para rodar em ambiente sem interface (sem mostrar os gráficos) | 1 para TREINO em S.O. com interface.

## Execução de Dados

### Rodar Dados em um Modelo

Para isso, precisa-se de:
- Um modelo já treinado que esteja dentro de `modelos/`;
- De um dataset de dados do hotfilm em tensão (preparado a seguir).

1. **Organização dos arquivos**:
    Após a coleta dos dados de tensão do hotfilm em 2kHz em `.csv`, crie uma pasta única no formato `collected_data_xxx` dentro do diretório:
    ```
    data/raw_data/raw_run
    ```

2. **Estrutura da pasta**:
    Depois de criada, a pasta deve conter o dataset `hotfilm_xyz.csv` da seguinte forma:
    ```
    data/raw_data/raw_run/collected_data_xyz/hotfilm_xyz.csv
    ```

    **Exemplos**:
    ```
    collected_data_xyz
    └── hotfilm_xyz.csv
    ```

3. **Preparação dos dados**:
    Execute o seguinte comando para preparar os dados com conhecimento do identificador:
    ```bash
    $ python3 csv_maker.py run {serie}
    ```
    **Exemplo**:
    ```bash
    $ python3 csv_maker.py run xyz
    ```
    Será salvo um arquivo único `run_xyz.csv` dentro do diretório `data/run`.

4. **Processamento dos dados**:
    Após essa preparação dos dados, execute o seguinte comando para processar os dados dentro da rede neural:
    ```bash
    $ python3 run_mlp.py {serie} {nome do modelo salvo}.pth
    ```
    **Exemplo**:
    ```bash
    $ python3 run_mlp.py xyz model_XXX.pth
    ```
    **Obs:** Não esqueça de colocar o nome do modelo idêntico ao encontrado dentro da pasta `modelos/` junto de sua extensão `.pth`.
    O resultado da execução será salvo dentro de uma pasta (`velocity_xyz`) com o mesmo identificador de série em:
    ```
    data/run/run_results
    ```

### Observações

- O modelo deve estar dentro da pasta `modelos` ou abaixo (há de se especificar no argumento o subdiretório).
- Os dados de entrada têm de estar em `.csv` do seguinte formato: `| time, voltage_x, voltage_y, voltage_z |`.
- Os dados salvos serão encontrados em `data/gerados/` em uma pasta específica desse processamento com o identificador de série.

## Geração do Espectro e Análise da Dissipação ($\epsilon$)

O script `spectrum.py` é utilizado para gerar o gráfico da Densidade Espectral de Potência (DEP) da velocidade predita e do sônico (para comparação) e, opcionalmente, calcular a **Taxa de Dissipação de Energia Cinética da Turbulência ($\epsilon$)** e validar o código de integração contra modelos teóricos.

### 1\. Requisitos Adicionais de Dados

Para que o `spectrum.py` execute os cálculos de Dissipação ($\epsilon$) e Validação, dois arquivos adicionais, contendo constantes e dados teóricos, são necessários:

#### a. Arquivo de Configuração (JSON)

Este arquivo armazena constantes físicas e parâmetros de cálculo específicos para a série, permitindo que o código rode sem modificar as variáveis internas.

1.  Crie uma pasta `config` dentro de `data/`.
2.  O arquivo deve ser nomeado no formato `config_{SERIE}.json` e salvo em:
    ```
    data/config/config_{SERIE}.json
    ```

**Estrutura de Exemplo (`config_xxx.json`):**

```json
{
    "KINEMATIC_VISCOSITY": 1.516e-5,    // Viscosidade Cinemática ($\nu$)
    "FS_HOTFILM": 2000,                 // Frequência de Amostragem do Hot-film (Hz)
    "FS_SONIC": 20,                     // Frequência de Amostragem do Sônico (Hz)
    "EPSILON_EXPECTED": 0.0106,         // Valor de $\epsilon$ esperado para a condição da série
    "THEORETICAL_MODEL": {
        "PATH": "data/raw_data/models_spec5940.dat", // Caminho para o arquivo teórico
        "E11_COLUMN": 1,                    // Índice da coluna do espectro longitudinal E11 (e.g., Meyers-Meneveau)
        "E_TRANS_COLUMN": 2                 // Índice da coluna do espectro transversal E_trans (e.g., Meyers-Meneveau)
    }
}
```

#### b. Arquivo de Espectro Teórico (Para Validação)

Este arquivo, mencionado em `THEORETICAL_MODEL:PATH` no JSON, contém dados de espectro gerados por modelos matemáticos, essenciais para validar a precisão da sua função de integração de $\epsilon$.

1.  O arquivo (ex: `models_spec5940.dat`) deve conter colunas separadas por espaço.
2.  A **primeira coluna** deve ser o **Número de Onda ($k$)**.
3.  As colunas subsequentes devem ser os valores de $E(k)$.

### 2\. Execução e Opções de Análise

1.  Execute o comando a seguir, fornecendo o identificador da série:

    ```bash
    $ python3 spectrum.py {SERIE}
    ```

    **Exemplo**:

    ```bash
    $ python3 spectrum.py xxx
    ```

2.  Após o início, o script apresentará duas perguntas que controlam os cálculos:

    | Pergunta | Cálculo Relacionado | Necessidade do JSON |
    | :--- | :--- | :--- |
    | **"Você deseja calcular a Taxa de Dissipação ($\epsilon$) a partir do modelo predito?"** | Calcula a $\epsilon$ utilizando os espectros suavizados da série predita (o resultado do seu modelo RN). | **Sim**, usa todas as constantes do JSON. |
    | **"Você deseja realizar o Teste de Validação Teórica?"** | Verifica a precisão da função de integração, usando o arquivo teórico (Seção 1b). | **Sim**, usa $\epsilon_{EXPECTED}$ e `THEORETICAL_MODEL` do JSON. |

### 3\. Detalhes da Geração do Espectro

  * **Periodograma Plotado:** O gráfico exibido utiliza o **espectro bruto** (não suavizado) do sinal de flutuações, para garantir a visualização direta de todo o ruído e estrutura do sinal.
  * **Cálculo da Dissipação ($\epsilon$):** A taxa $\epsilon$ é calculada apenas se o usuário optar por isso. A integração é feita sobre os espectros das três componentes de velocidade ($E_{11}, E_{22}, E_{33}$), que são previamente **suavizados por média em bins logarítmicos** para estabilizar o cálculo numérico. O resultado é comparado com o `EPSILON_EXPECTED` do arquivo JSON.