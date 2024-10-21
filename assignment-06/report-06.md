## IDATT2502 - Exercise 06 - Dimension Reduction in Weka

[Weka](https://waikato.github.io/weka-wiki/)

### Part 1

> Using the UCI mushroom dataset from the last exercise, perform a feature selection using a classifier evaluator. Which features are most discriminitave?

Weka output: 

```
=== Attribute Selection on all input data ===

Search Method:
	Best first.
	Start set: no attributes
	Search direction: forward
	Stale search after 5 node expansions
	Total number of subsets evaluated: 167
	Merit of best subset found:    0.67 

Attribute Subset Evaluator (supervised, Class (nominal): 23 habitat):
	Classifier Subset Evaluator
	Learning scheme: weka.classifiers.rules.DecisionTable
	Scheme options: -X 1 -S weka.attributeSelection.BestFirst -D 1 -N 5 
	Hold out/test set: Training data
	Subset evaluation: classification error

Selected attributes: 12,19,21,22 : 4
                     stalk-root
                     ring-number
                     spore-print-color
                     population
```



### Part 2

> Use principal components analysis to construct a reduced space. Which combination of features explain the most variance in the dataset?

PCA, Weka output:

```
Ranked attributes:
 0.906      1 0.248stalk-surface-above-ring=k+0.247stalk-surface-below-ring=k−0.246edibility=e+0.224odor=f+0.218ring-type=l...
 0.8358     2 0.242gill-attachment=a−0.238veil-color=w+0.238stalk-color-above-ring=o+0.238stalk-color-below-ring=o+0.224habitat=l...
 0.772      3 −0.242ring-type=e−0.236gill-color=b+0.236gill-attachment=a−0.231stalk-shape=t+0.23 gill-size=b...
```



### Part 3

> Do you see any overlap between the PCA features and those obtained from feature selection?

There are no clear overlaps between the PCA features and those obtained from feature selection.
