### Comparação de modelos e transformação de dados para modelagem de crédito (Give Me Some Credit Kaggle)

O objetivo deste estudo foi comparar quatro abordagens distintas para modelagem de crédito, considerando bases balanceados, desbalanceadas, variáveis transformadas por WOE, bem como em sua escala original. No processo de modelagem foram utilizados os algoritmos de Random Forest, Redes Neurais, AdaBoost, Gradient Boosting e Regressão Logística, em cada um destes casos foram realizadas permutações nos hiperparâmetros dos modelos(tuning), com o intuito de selecionar o mais adequado para cada um dos quatro cenários.

A base utilizada no estudo pode ser obtida no site Kaggle (https://www.kaggle.com/c/GiveMeSomeCredit) e conta principalmente com variáveis relacionadas ao comportamento do indivíduo no crédito. A variável target é uma dummy, que indica os casos que atingiram atrasos superiores a 90 dias.

### Análise descritiva
<img align="center" width="600" height="360"  src="https://github.com/WOLFurriell/BehaviorScorePython/blob/master/plots/donut3.png">

Pelo gráfico exposto nota-se que a distribuição da variável target é desbalanceada, deste modo, foi realizada um undersampling na base, para balancear a amostra. 
Tal método equaliza a informação desbalanceada diminuindo de forma aleatória o conjunto com a classificação majoritária. Tendo em vista tal aspecto, neste estudo realizou-se um balanceamento 66.6/33.3.

<img align="center" width="800" height="350"  src="https://github.com/WOLFurriell/BehaviorScorePython/blob/master/plots/balanc2.png">

No gráfico acima pode-se observar os resultados da variável target para a base balanceada e desbalanceada.

<table><thead><tr><th>Volume de missings</th><th>f</th><th>%</th></tr></thead><tbody><tr><td>Perf_final :</td><td>0</td><td>0</td></tr><tr><td>UltPercLimit :</td><td>0</td><td>0</td></tr><tr><td>Idade :</td><td>0</td><td>0</td></tr><tr><td>N_Atraso30_59Dias :</td><td>0</td><td>0</td></tr><tr><td>RazaoGastos :</td><td>0</td><td>0</td></tr><tr><td>RendaMensal :</td><td>17753</td><td>20</td></tr><tr><td>N_EmeprestimosAbertos :</td><td>0</td><td>0</td></tr><tr><td>N_atrasos_Ult90Dias :</td><td>0</td><td>0</td></tr><tr><td>N_emprestimos :</td><td>0</td><td>0</td></tr><tr><td>N_Atraso60_89Dias :</td><td>0</td><td>0</td></tr><tr><td>N_dependentes :</td><td>2300</td><td>3</td></tr></tbody></table>

<table><thead><tr><th>Percentis</th><th>UltPercLimit</th><th>Idade</th><th>N_Atraso30_59Dias</th><th>RazaoGastos</th><th>RendaMensal</th><th>N_EmeprestimosAbertos</th><th>N_atrasos_Ult90Dias</th><th>N_emprestimos</th><th>N_Atraso60_89Dias</th><th>N_dependentes</th></tr></thead><tbody><tr><td>0.10</td><td>0.002969</td><td>33.0</td><td>0.0</td><td>0.030874</td><td>2005.0</td><td>3.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>0.50</td><td>0.154181</td><td>52.0</td><td>0.0</td><td>0.366508</td><td>5400.0</td><td>8.0</td><td>0.0</td><td>1.0</td><td>0.0</td><td>0.0</td></tr><tr><td>0.80</td><td>0.698857</td><td>65.0</td><td>0.0</td><td>4.000000</td><td>9083.0</td><td>12.0</td><td>0.0</td><td>2.0</td><td>0.0</td><td>2.0</td></tr><tr><td>0.85</td><td>0.854021</td><td>68.0</td><td>1.0</td><td>269.150000</td><td>10100.0</td><td>13.0</td><td>0.0</td><td>2.0</td><td>0.0</td><td>2.0</td></tr><tr><td>0.90</td><td>0.981278</td><td>72.0</td><td>1.0</td><td>1267.000000</td><td>11666.0</td><td>15.0</td><td>0.0</td><td>2.0</td><td>0.0</td><td>2.0</td></tr><tr><td>0.95</td><td>1.000000</td><td>78.0</td><td>2.0</td><td>2449.000000</td><td>14587.6</td><td>18.0</td><td>1.0</td><td>3.0</td><td>1.0</td><td>3.0</td></tr><tr><td>0.99</td><td>1.092956</td><td>87.0</td><td>4.0</td><td>4979.040000</td><td>25000.0</td><td>24.0</td><td>3.0</td><td>4.0</td><td>2.0</td><td>4.0</td></tr><tr><td>0.99</td><td>1.092956</td><td>87.0</td><td>4.0</td><td>4979.040000</td><td>25000.0</td><td>24.0</td><td>3.0</td><td>4.0</td><td>2.0</td><td>4.0</td></tr><tr><td>1.00</td><td>50708.000000</td><td>109.0</td><td>98.0</td><td>329664.000000</td><td>3008750.0</td><td>58.0</td><td>98.0</td><td>54.0</td><td>98.0</td><td>20.0</td></tr></tbody></table>

Dois pontos importantes a serem considerados no processo de modelagem são a avaliação dos missings e a identificação de outliers. Neste caso, verifica-se que a variável RendaMensal, possui 20% de missings, logo um estudo de imputação de dados pode ser considerado para o prenchimento dos casos faltantes. Acerca disto, meu amigo Victor, tem um artigo bastante interessante em seu blog: https://nagahamavh.github.io/posts/data-understanding-in-data-science/.
Além disso, nota-se a existência de pontos influentes a partir do percentil 99. Desse modo, foi realizada a substituição das informações acima desta métrica pelo P99. 

<img align="center" width="800" height="500"  src="https://github.com/WOLFurriell/BehaviorScorePython/blob/master/plots/hist3.png">

Antes de iniciar o processo de modelagem foi realizada uma breve análise univariada das informações, tendo por objetivo avaliar a distribuição dos dados. Partindo deste diagnóstico foi aplicado o logaritmo natural para suavizar a distribuição da RazaoGastos, dado sua assimetria extrema. 

<img align="center" width="800" height="500"  src="https://github.com/WOLFurriell/BehaviorScorePython/blob/master/plots/Box3.png">

No gráfico acima tem-se os box-plots das variáveis preditoras segundo a target, nos quais, é possível avaliar descritivamente que existem distinções entre as médias das informações quando comparados os indivíduos bons e maus.

<img align="center" width="600" height="400"  src="https://github.com/WOLFurriell/BehaviorScorePython/blob/master/plots/cor1.png">

Para encerrar a parte descritiva, tem-se o gráfico de correlação das variáveis preditoras. Avaliando seus resultados não foram constatados graus de correlação que representassem problemas para os modelos de regressão. Ressalta-se que foi aplicado a correlação de Pearson para informações contínuas e Spearman para as discretas.

### Peso de evidência e valor da informação (WOE e IV)

Uma prática bastante comum na modelagem de crédito é realizar a transformação das variáveis por WOE, tal método elimina a influência de outliers e permite uma pré seleção das informações, considerando uma relação "linear" com a variável target e uma avaliação da capacidade de discriminatória com o IV (Information Value). Para o cálculo do WOE é preciso categorizar as variáveis caso sejam contínuas. O WOE e IV são dados por:

<a href="https://www.codecogs.com/eqnedit.php?latex=WOE&space;=&space;ln\left&space;(&space;\frac{%Bons}{%Maus}\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?WOE&space;=&space;ln\left&space;(&space;\frac{%Bons}{%Maus}\right&space;)" title="WOE = ln\left ( \frac{%Bons}{%Maus}\right )" /></a>, 

<a href="https://www.codecogs.com/eqnedit.php?latex=IV=&space;\sum&space;\left&space;(&space;{%Bons}-{%Maus}\right&space;).WOE" target="_blank"><img src="https://latex.codecogs.com/gif.latex?IV=&space;\sum&space;\left&space;(&space;{%Bons}-{%Maus}\right&space;).WOE" title="IV= \sum \left ( {%Bons}-{%Maus}\right ).WOE" /></a>

Considera-se que valores de IV < 0.1, indicam baixa capacidade de discriminação entre eventos e não eventos, entre 0.1 e 0.3 média e maior que 0.3 alta. É válido ressaltar que as categorias das variáveis para aplicação do WOE devem ter pelo menos 5% da distribuição geral dos dados e volume diferente de zero tanto para eventos como não eventos.

<img align="center" width="800" height="1300"  src="https://github.com/WOLFurriell/BehaviorScorePython/blob/master/plots/WOE4.png">

Com os resultados do WOE nota-se que a variável RazaoGastos apresenta uma inversão, isto é, não segue uma relação "linear" com a target, deste modo, para garantir a interpretabilidade dos resultados dos modelos, principalmente quando se trata da Regressão Logística, não a consideramos na análise. Uma alternativa para considerar variáveis categóricas ou categorizadas, não seguem ordenação seria utilizá-las como dummy, contudo, tal abordagem é interessante desde que, a variável selecionada tenha explicação prática. 
Destaca-se que as preditoras que apresentaram maiores IV foram N_Atraso60_89Dias, N_atrasos_Ult90Dias e CAT_UltPercLimit.

### Resultados dos modelos de classificação

Na tabela abaixo é possível verificar um resumo dos modelos empregados.

<table><thead><tr><th></th><th>WOE <br>Balanceado</th><th>WOE<br> Desbalanceado</th><th>Contínuas <br>Balanceado</th><th>Contínuas<br>Desbalanceado</th></tr></thead><tbody><tr><td>1</td><td colspan="4">Regressão Logística </td></tr><tr><td>2</td><td colspan="4">Random Forest</td></tr><tr><td>3</td><td colspan="4">Gradiet boosting</td></tr><tr><td>4</td><td colspan="4">AdaBoost</td></tr><tr><td>5</td><td colspan="4">Redes Neurais</td></tr></tbody></table>

Para avaliar a ordenação da predição segundo a resposta observada, isto é, se as predições dos indivíduos maus dividida em dez faixas seguem a proporção de maus observados dentro das mesmas. Foi ultilizada a medida de Lift, dada por:

<a href="https://www.codecogs.com/eqnedit.php?latex=Lift_{fx}&space;=&space;\frac{%Maus_{fx}}{%Maus_{obs}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Lift_{fx}&space;=&space;\frac{%Maus_{fx}}{%Maus_{obs}}" title="Lift = \frac{%Maus_{fx}}{%Maus_{obs}}" /></a>

<img align="center" width="800" height="700"  src="https://github.com/WOLFurriell/BehaviorScorePython/blob/master/plots/ordena4.png">

Pelo gráfico acima, constata-se que todos os modelos apresentaram um ajuste satisfatório, sem inversões acentuadas no decorrer das curvas. Ou seja, conforma aumenta nossa probabilidade de maus predita, também aumenta a porcentagem de maus observada.

<img align="center" width="800" height="2100"  src="https://github.com/WOLFurriell/BehaviorScorePython/blob/master/plots/ranks5.png">

Quanto as medidas de avaliação, no estudo foram considerados a Acurácia, Precisão, Recall, LogLoss, AUC, KS, Gini, ASE e Shift AUC. Dessa forma, constatou-se que os modelos de Gradient Boosting e Radom Forest para bases desbalanceadas e variáveis contínuas foram os que melhor performaram. Contudo, destaca-se que todos os modelos apresentaram ajustes safisfatórios, considerando a medidade de AUC os resultados foram em torno de 80.

<img align="center" width="900" height="400"  src="https://github.com/WOLFurriell/BehaviorScorePython/blob/master/plots/compara.png">

Por fim, selecionou-se as medidas de Acurácia, AUC, KS, LogLoss e Gini e foi aplicada a padronização MinMáx para mantê-las no mesmo intervalo de 0 a 1 e com o mesmo peso. Somando estes resultados, verificou-se que o modelo que mais pontuou foi o Random Forest com dados desbalanceados e variáveis contínuas.

