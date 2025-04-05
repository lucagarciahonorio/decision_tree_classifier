# 📊 Comparação de Árvore de Decisão: Scikit-learn vs Implementação Manual

Este projeto tem como objetivo comparar a performance e funcionamento de uma **árvore de decisão** construída com a biblioteca **Scikit-learn** com uma **implementação manual feita em Python puro**, utilizando o famoso dataset **Iris**.

---

## 🧠 Objetivo

Demonstrar, de forma didática, como funciona o algoritmo de árvore de decisão na prática e como sua performance pode variar ao utilizar uma biblioteca especializada como o Scikit-learn versus uma implementação do zero.

---

## 🗂️ Estrutura do Projeto

decision_tree_comparison/ ├── decision_tree_comparison.py # Script principal do projeto ├── README.md # Este arquivo └── requirements.txt # Bibliotecas necessárias

yaml
Copiar
Editar

---

## 🧪 Base de Dados

Utiliza-se o clássico dataset **Iris** do `sklearn.datasets`, que possui 150 amostras de flores classificadas em 3 espécies com 4 atributos cada (largura e comprimento das pétalas e sépalas).

---

## 🚀 Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/seuusuario/decision_tree_comparison.git
cd decision_tree_comparison
2. Instale as dependências
bash
Copiar
Editar
pip install -r requirements.txt
3. Execute o script
bash
Copiar
Editar
python decision_tree_comparison.py
✅ Resultado Esperado
Ao final da execução, o script exibirá as acurácias dos dois modelos:

bash
Copiar
Editar
Acurácia (Scikit-learn): 1.00
Acurácia (Árvore feita na mão): 0.91
A diferença ocorre pois a versão manual é mais simples e não possui técnicas avançadas como poda, controle de overfitting, etc.

📚 Conceitos Abordados
Entropia e ganho de informação

Implementação recursiva de árvore de decisão

Comparação de desempenho entre bibliotecas e código puro

Classificação supervisionada com dados reais

🛠️ Tecnologias Utilizadas
Python 3.8+

Scikit-learn

Pandas

NumPy

📄 Licença
Este projeto está licenciado sob a MIT License.

🤝 Contribuição
Contribuições são bem-vindas! Sinta-se livre para abrir issues ou pull requests.

yaml
Copiar
Editar

---

Se quiser, posso gerar o conteúdo do `requirements.txt` também. Deseja?







