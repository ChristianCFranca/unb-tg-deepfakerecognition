# unb-tg-deepfakerecognition

Este é o repositório oficial para o meu trabalho de conclusão de curso `Detectando Deepfakes em vídeos: Uma abordagem utilizando redes neurais convolucionais residuais profundas` da Universidade de Brasília como etapa final na obtenção do meu diploma de Engenheiro de Controle e Automação.

Os principais arquivos a serem recorridos são os arquivos jupyter notebook (`.ipynb`).

## CreateImagesDataset.ipynb

Este caderno contém o passo-a-passo para a criação do conjunto de dados de imagens a partir do conjunto de dados de vídeos [_Deepfake Detection Challenge_](https://www.kaggle.com/c/deepfake-detection-challenge).

## Treinamento Fastai Resnet18.ipynb

Este caderno contém todo o processo de treinamento do modelo `Resnet18` em validação cruzada e depois no conjunto de treinamento. Todas as métricas de cada uma das etapas também está presente nesse arquivo. Ele faz uso do conjunto de dados de imagens criado anteriormente pelo arquivo `CreateImagesDataset.ipynb`.

## End-to-end.ipynb

Este caderno contém a implementação de uma simples solução end-to-end onde um vídeo entra e a classificação final para ele é gerada. O objetivo é averiguar o tempo de computação necessário para o processo completo desde a extração dos rostos de um vídeo até a geração da predição final.
