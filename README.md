# Tents and Trees.
Using linear programming to solve the tents and trees game often found in newspapers. Tents and trees is a game where tents have to be placed on a campsite by following these rules:

- The numbers outside the grid show how many tents in that row or column.
- Tents can only be planted next to trees (horizontally or vetrically)
- Tents do not touch each other, not even diagonally.

The repository contains a [notebook](https://github.com/rrodenburg/tents_n_trees/blob/master/tents_n_trees.ipynb) in which we solve the following example using linear programming in python:

![alt text](https://github.com/rrodenburg/tents_n_trees/blob/master/example.png "Example of a Tents and trees game")


## Solution
The first column and bottom row represent the number of tents required in each row of column. Trees are represented by 'T', placed tents are indicated with '1'.

![alt text](https://github.com/rrodenburg/tents_n_trees/blob/master/solution.png "Solution of the example")
