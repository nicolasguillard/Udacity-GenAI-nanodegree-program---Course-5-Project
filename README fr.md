# HomeMatch : exploitation de llm et base de données vectorielle pour la génération et la recherche d'annonces immobilières

## Description
Ce projet exploite à travers la librairie `langchain` un LLM pour générer des annonces immobilières comprenant des descriptions de maison, et pour l'affichage de résultats optimisés selon la demande de l'utilisateur.

## Installation
Après récupération du projet, il est nécessaire de créer un environnement virtuel et d'installer les dépendances requises.
```bash
pip install -r requirements.txt
```

Il faut ensuite créer un fichier `.env` à la racine du projet contenant les variables d'environnement suivantes :
```bash
OPENAI_API_KEY=...
OPENAI_API_BASE=https://...
```

## Utilisation
D'abord via le carnet Jupyter python `HomeMatch.ipynb`, pour générer les annonces immobilières et les stocker dans une base de données vectorielle, puis tester les fonctionnalités et d'augmentation d'annonce.

Les annonces générées sont également stockées dans un fichier txt `listings.txt` au format JSON (et aussi dans `listings.jsonl`), pour une utilisation ultérieure (dont avec `HomeMatch.py`).

Via l'application reposant sur gradio afin de disposer d'une interface utilisaeur pour effectuer des recherches sur la base de données vectorielle et d'afficher les résultats :
```bash
python HomeMatch.py
```

![HomeMatch Gradio User Interface](https://github.com/nicolasguillard/Udacity-GenIA-nanodegree-program---Course-5-Project/blob/main/homematch_gradio_screenshot.png?raw=true)


En dehors de l'interface utilisateur, ce script reprend tout le code relatif aux fonctionnalités de recherche et d'augmentation d'annonces présent dans le carnet Jupyter `HomeMatch.ipynb`.
