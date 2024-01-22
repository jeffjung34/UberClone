# Overview:

This GitHub repository hosts a comprehensive set of algorithms designed to efficiently match passengers with drivers. The project's core focus is on optimizing various aspects of ride-sharing, such as wait time, distance, and overall system efficiency. 

Each file attemps to optimize the following contraints:

### D1. Passengers want to be dropped off as soon as possible, that is, to minimize the amount
of time (in minutes) between when they appear as an unmatched passenger, and when they
are dropped off at their destination.
### D2. Drivers want to maximize ride profit, defined as the number of minutes they spend
driving passengers from pickup to drop-off locations minus the number of minutes they spend
driving to pickup passengers. For simplicity, there is no penalty for the time when a driver is
idle.
### D3. Your algorithms should be empirically efficient and scalable.

# Files Description
## T1 - Baseline Queue Matching Algorithm
T1 implements a baseline algorithm that matches the longest waiting passenger with the first available driver, based on simple queue management. This file includes an experiment evaluating the algorithm's performance against key metrics (D1-D3).
## T2 - Distance-Based Matching Algorithm
T2 enhances the baseline by matching drivers and passengers based on the minimum straight-line distance. This file contains the implementation and an experiment evaluating its performance improvements.
## T3 - Time-Efficient Matching Algorithm
T3 introduces an algorithm focusing on minimizing the estimated time for a driver to reach a passenger. It includes a detailed explanation of the algorithmic approach and an experiment assessing its effectiveness against the desiderata D1-D3.
## T4 - Optimized Time-Efficient Matching Algorithm
T4 further optimizes the time efficiency of the T3 algorithm. It explores preprocessing techniques for road network nodes and an advanced algorithm for computing shortest paths. The file includes experiments demonstrating the empirical correctness of the optimizations and their impact on query runtimes.
## T5 - Innovative Scheduling Algorithm
T5 features a novel scheduling algorithm developed to surpass the baseline models on one or more desiderata (D1-D3), or other relevant metrics. This file details the algorithm, its implementation, and experiments showcasing its improvements.
## Case Study
The Case Study PDF is a comprehensive document summarizing the findings from all the experiments conducted. It offers insights into the effectiveness of each algorithm and their comparative analysis.
# Getting Started
To get started with the NotUber project, clone the repository and navigate to each file (T1-T5) to understand the individual algorithms and their implementations. The case study PDF provides a holistic view of the project's findings and should be referred to for an in-depth understanding of the outcome
