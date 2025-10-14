# Manual de Instalação

1. Instale o ambiente virtual do Python em sua máquina:
    ```bash
    $ pip install virtualenv
    ```

2. Vá para uma pasta que deseja que esteja o workspace do projeto e copie o projeto para a pasta local, usando o `git clone`, por exemplo:
    ```bash
    $ git clone https://github.com/LucasDuarte026/hotfilm_NN.git
    ```

3. Dentro da pasta `./hotfilm_NN`, execute o seguinte comando:
    ```bash
    $ virtualenv hotfilm_env
    ```

4. Entre em modo ambiente virtual da seguinte forma:
    ```bash
    $ source hotfilm_env/bin/activate
    ```
    Para garantir que o procedimento foi sucedido, deve aparecer `(hotfilm_env)` ao começo da linha de seu shell de preferência, indicando que se está dentro do ambiente virtual.
    **Obs:** Caso queira sair do ambiente, execute:
    ```bash
    $ deactivate
    ```

5. Instale todas as dependências necessárias dos algoritmos de treinamento e de execução da rede:
    Antes: certifique-se que esteja dentro do environment (Passo 4).
    ```bash
    $ pip install -r requirements.txt
    ```

Neste momento, todas as dependências do código serão instaladas na máquina e poderão ser utilizadas na Rede Neural. Prossiga com o `manual.md` para execução dos algoritmos propriamente.
