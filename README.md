# PROJET MACHINE LEARNING :


**Introduction** :

En l'espace de plusieurs dizaines d'années les prix des joueurs de football ont explosé. Avec cette **nette augmentation** du prix des joueurs, il nous est **difficile** d'établir **le prix d'un joueur** actuellement.

Il serait intéressant de pouvoir prédire la valeur d'un joueur en fonction de son **rendement** sur le terrain. C'est ce que nous allons faire dans cet article. Nous disposons d'une base de données qui vient de **Wyscout**.

Dans ce dataset, nous avons récupérer des variables de **performance offensive** de joueur évoluant au poste de **buteur**. Nous allons donc construire un modèle de **régression linéaire** pour tenter de prédire au mieux la **valeur marchande** d'un joueur.


## Choix du modèle :

Puisque nous cherchons à **prédire** un prix, il semblerait que la **régression** soit effectivement le modèle le plus adapté.


Une régression linéaire doit respecter 3 hypothèses :

* **Linéarité** : Il faut que votre dataset ait une évolution linéaire
* **Homoscedasticité** : La variance de votre dataset ne doit pas être trop forte
* **Non-colinéarité** : Il faut que les variables prédictives n'aient pas de relation forte entre elles​

Une fois qu'on pense avoir un dataset ayant ces **3 hypothèses** on peut alors faire notre régression linéaire à variables multiples.



## Importer les librairies et notre dataset sur Python :

Pour cette prédiction, nous aurons besoin des **librairies** classiques Numpy, Pandas, Matplotlib. Nous importerons scikitlearn plus tard une fois que nous commencerons à nettoyer nos données puis appliquer notre modèle.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
**Importons maintenant notre dataset** :

```
dataset = pd.read_csv("players.data1.csv")
```

Nous avons des valeurs **numériques** mais aucune valeurs **catégoriques**. Cela nous évitera de les encoder pour les incorporer au modèle.


Visualisation de nos variables numériques :

Pour voir si notre modèle de régression peut effectivement fonctionner, regardons d'abord la **corrélation** qu'il y a entre quelques variables numériques et la valeur marchande de nos joueurs.

On commence par la plus évidente : le nombres de buts par 90 min (match) inscrit par le joueur.

```
sns.lmplot(x = "Buts par 90 min", y = "Valeur marchande", data = dataset, logistic = False)
```

A première vue, on peut observer que le nombre de buts inscrit par match fait **augmenter** la valeur d'un joueur.

Regardons une deuxième variable : le nombres de **passes décisives** par 90 min.

```
sns.lmplot(x = "Buts par 90 min", y = "Âge", data = dataset, logistic = False)
```

Ici c'est un peu plus difficile d'observer une linéarité entre le nombre de passes décisives et la valeur marchande d'un joueur.
Cependant, on constate tout de même une petite corrélation entre la valeur d'un joueur et son taux de passes décisives par match.

Nous avons quelques données qui sortent du lots et qui influent donc sur la linéarité.

**Observons une dernière variable** : le taux de conversion but/tir

Ici, nous constatons clairement qu'il n'y a **aucune** linéarité entre le taux de conversion but/tir et la valeur marchande d'un buteur. En effet, la valeur marchande d'un attaquant ne varie pas en fonction de son taux de conversion but/tir.

Cette variable là est donc une variable qui nous ne permettra pas de prédire le prix de nos joueurs.


D'après les trois variables que nous avons regardées, on peut dire qu'il y a une corrélation qui n'est pas exactement linéaire entre les variables et la valeur marchande des joueurs. Certaines ne le sont pas du tout même.

Cependant, il y a tout de même une certaine **homoscédascité** puisque, malgré quelques **outlayers**, les points ne sont pas trop éloignés les uns des autres.

Notre modèle de régression linéaire ne va donc pas être parfait mais il va pouvoir déjà nous donner une bonne vision d'ensemble et des prédictions qui ne seront pas si éloignées que ça de la réalité.


## Mise en place du modèle :

Séparation des variables **indépendantes** et la variable **dépendante** :

```
X = dataset.iloc[:,[0, 2, 3, 4]]
y = dataset.iloc[:,1]
```

Nous avons peu de variable indépendantes donc il est facile de visualiser notre dataset sans faire une ligne de code pour chercher les valeurs manquantes. Nous n'avons **aucune** valeurs manquantes dans notre dataset.

Application de notre modèle de
régression linéaire :

Notre phase de data **preprocessing** est maintenant terminée. Nous pouvons donc séparer notre dataset en **training set** et un **test set**. Nous choisirons un ratio de **80 / 20** pour séparer nos données.

```
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)
```

**Appliquons** maintenant notre modèle de régression linéaire sur notre training set et faisons notre première **prédiction** sur X_test :

```
from sklearn.linear_model import LinearRegression
regressor_lr = LinearRegression()
regressor_lr.fit(X_train,y_train)

y_pred_lr = regressor_lr.predict(X_test)
```

Tout fonctionne bien. Regardons les premières prédictions de notre modèle par rapport à nos valeurs test.

```
# Prediction the test results
y_pred = regressor.predict(X_test)

# Scoring Mean Square ERROR
ecart = ((y_pred - y_test)**2)**(1/2)
ecart.mean()


overview_y_pred = y_pred_lr
overview_y_test = y_test


overview = pd.DataFrame(data=np.column_stack((overview_y_pred,overview_y_test)),
                        columns=["Pred", "Valeurs réelles"])

overview.head()
```


Il semblerait que notre modèle fasse du plutôt bon travail hormis pour la cinquième valeur pour laquelle nous sommes de 20Meuros éloignés.




## Évaluation de notre modèle :

Une manière assez simple d'évaluer notre modèle serait de voir **l'écart moyen** entre nos valeurs prédites et nos valeurs réelles. De cette manière, nous aurons une vue globale de la performance de notre modèle.

**Scoring Mean Square ERROR**

```
ecart = ((y_pred_lr - y_test)**2)**(1/2)
ecart.mean()
Out[242]: 4435121.7289730925
```
Nous pouvons voir ici que nous avons un écart moyen entre nos valeurs réelles et nos valeurs prédites de **4,4M d'euros environ**. Ce qui correspond à **15% d'écart** environs. Ce qui est correct pour un modèle de régression linéaire simple.

Pistes d'**amélioration** du système :

Il est toujours possible d'améliorer un modèle.

Voici quelques **idées** qui pourraient faire la différence :

* Regarder la **co-linéarité** entre les variables et retirer celles qui ont une relation trop forte
* Ajouter de **nouvelles variables** pour affiner le modèle
* Supprimer les **outlayers** qui faussent le modèle de prédiction

Ces idées seront donc à développer pour optimiser nos prédictions.
