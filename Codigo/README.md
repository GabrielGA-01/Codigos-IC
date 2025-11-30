O que foi feito a partir do dia 25

- Correção do tratamendo dos dados dos datasets para que eles tratem corretamente a nova variável sensível em duas classes. As instâncias com mais de 60 anos foram removidas.
- Generalização da função smote para funcionar com qualquer variável sensível e para quaisquer duas classes.
- Separação do notebook em vários códigos para facilitar a visualização, pois havia muito código no notebook e era difícil chegar ao ponto desejado.
- Atualização das métricas de desempenho, adicionado a métrica f1 e KS, além de uma correção das métricas de fairness nos casos em que ocorria uma divisão por zero (especialmente no dataset 2) para lidarem com a exceção corretamente. Ademais, o cálculo das métricas de fairness foi atualizado para que a função possa surtar variar a variável sensível.
- Renomeação das variáveis sensíveis para ficar com o mesmo padrão dos notebooks da Julia.