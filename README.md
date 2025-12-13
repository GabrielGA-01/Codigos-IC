Repositório para o desenvolvimento do projeto Serasa DataLab.

O principal está todo dentro da pasta 'Codigo', separado em vários arquivos python. Cada arquivo tem uma função específica. Em alguns dos arquivos há exemplos de uso/teste que podem ser feitos ao executar o respectivo arquivo. Além disso, eles seguem a seguinte lógica:

- c0: variáveis auxiliares
- c1: funções de modo geral que não entram nas outras categorias
- c2: algoritmos de machine learning
- c3: técnicas de pré-processamento
- c4: técnincas de pós-processamento

- d0: funções específicas que não são usadas pelo restante do código
- d1: testes manuais (é preciso alterar os valores usados manualmente)

- z: funções para comparação do desempenho das técnicas a partir das planilhas 

O notebook 'note_pre-processamento.ipynb' contém toda a lógica do tratamento dos datasets usados, presentes no diretório 'Codigo/Datasets/Originais', para a criação dos datasts tratados, salvos em 'Codigo/Datasets/Processados'.

O notebook 'note_aplicadao.ipynb' contém o código usado para testar as técnicas, algoritmos e datasets definidos no projeto, além de gerar a tabela de resultados e salvar o dicionário com os valores calculados.

Na pasta 'Restante' estão alguns dos artigos utilizados, mas não são todos. Além de algumas versões anteriores dos códigos desenvolvidos (V7 e V10). Neles o código está todo em um único notebook e não contém a maioria das implementações mais recentes. Ademais, há inda alguns arquivos .txt com a descrição de cada uma das sete técnicas de mitigação de viés utilizadas e os requirements das bibliotecas utilizadas para esse código funcionar 'requirements_Gabriel' e os utilizados pala Julia, usados para garantir que sejam usadas as mesmas versões por nós dois. Por fim, há também alguns dos resultados antigos obtidos pelos notebooks V7 e V9.

# ATENÇÃO: 
é importante para que o resultados sejam salvos que haja um diretório 'Codigo/Resultados'. Essa pasta não está presente no GitHub pois esta sendo salvo muitos arquivos grandes, especialmente os dicionários, sendo que são de resultados parciais e muitos casos de teste, então ela não está sendo commitada aqui.

O arquivo requirements presente na raiz do projeto tem os dois requirements mencionados anteriormente já mesclados, contendo tudo o que é preciso para executar todos os códigos. 

É necessário instalar algumas bibliotecas:

pip install aif360
pip install 'aif360[AdversarialDebiasing]'
pip install 'aif360[inFairness]'
pip install fairlearn
pip install --upgrade inFairness
pip install BlackBoxAuditing
pip install imblearn
pip install seaborn
pip install shap
pip install xgboost
pip install openpyxl
pip install ucimlrepo