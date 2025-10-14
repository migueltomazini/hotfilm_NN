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

## Geração do Espectro

1. Execute o comando a seguir para criar um gráfico com o espectro do sinal de velocidade gerado através do processamento de um conjunto de dados de tensão por um modelo.
    ```bash
    $ python3 spectrum.py {SERIE}
    ```
    **Exemplo**:
    ```bash
    $ python3 spectrum.py xxx
    ```

    **Obs:** O dataframe que será utilizado pelo espectro é o que tiver na pasta com o identificador de série relativo `{SERIE}` dentro da pasta.

    **Exemplo de arquivo que será buscado caso xxx seja o identificador**:
    ```
    /data/run/velocity_xxx/velocity_xxx.csv
    ```
