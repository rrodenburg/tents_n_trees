# Tents and Trees.
Tents and trees is a puzzle often found in newspapers. To solve the puzzle, tents have to be placed on a campsite by following these rules:

- The numbers outside the grid show how many tents in that row or column.
- Tents can only be planted next to trees (horizontally or vetrically)
- Tents do not touch each other, not even diagonally.

This repository contains 2 notebooks, [one](./puzzle_detector.ipynb) that automatically detects the squares of the puzzle and determines if the square contains a tree. See examples:

Puzzle | Detected Trees | Solution
:-:|:-:|:-:
<img src="examples/example1.png" heigth="250" width="250"/> | <img src="examples/example1_trees.png" heigth="250" width="250"/> | <img src="examples/example1_solution.png" heigth="250" width="250"/>
<img src="examples/example2.png" heigth="250" width="250"/> | <img src="examples/example2_trees.png" heigth="250" width="250"/> | <img src="examples/example2_solution.png" heigth="250" width="250"/>

<!-- The repository contains a [notebook](https://github.com/rrodenburg/tents_n_trees/blob/master/tents_n_trees.ipynb) in which we solve the following example using linear programming in python: -->

<!-- ![alt text](./examples/example1.png "Example of a Tents and trees game") -->

<!-- The only 

## Solution
The first column and bottom row represent the number of tents required in each row of column. Trees are represented by 'T', placed tents are indicated with '1'.

![alt text](./examples
/example1_solution.png "Solution of the example") -->
