# CILv2-weather-testing
# Motivation
This repo was developed as a portfolio project to explore some subjects I am interested in, mainly Explainable AI (XAI), Vision-Action models, and the validation of autonomous vehicles. 

## Introduction and Objectives
This analysis aims to evaluate the performance of the CILv2 [here](https://arxiv.org/pdf/2302.03198), an End-to-End Autonomous Driving Visual-Action model, to identify scenarios where a specific weather condition can lead to hazard.

Because this testing is scenario-based and exploratory, it is inherently open-ended with no explicit operational design domain (ODD) requirements or strict pass/fail criteria provided by the original developers. I utilize traffic conflict techniques to evaluate the model's safety through surrogate safety measures, specifically focusing on path adherence and control stability under visual degradation.

For this analysis to be valid, we rely on several foundational assumptions:
* **The Accident-Conflict Axiom:** Traffic conflicts (such as severe lane deviations or extreme jerk) originate from the same underlying failure mechanisms as actual traffic accidents.
* **The Constant Velocity Hypothesis:** When calculating predictive safety metrics like Time to Line Crossing (TLC), we assume the vehicle maintains its current longitudinal and lateral velocity over the immediate prediction horizon.
* **The Unchanged Trajectory Hypothesis:** We assume the vehicle's current steering angle and acceleration vectors remain constant until the lane boundary is breached, meaning the AI does not initiate an emergency evasive maneuver during the micro-calculation window.

## Design of Experiment and Sampling

### Why test?
* **Characterize**
  * Want to know the effect of each factor on the response and how the factors may interact with each other.
* **Predict**
  * Want to predict responses for a given level(s) of the factor(s).
* **Optimize**
  * Want to find the levels of the factors that optimizes the responses.
* **Design**
  * Want to identify key parameters, compare alternatives.
* *Note: The points – Optimize and Design – are out of the scope of this project but one aim of this project is to try to come up with recommendations based on the statistical and XAI analysis.*

### Environmental Isolation & Constraints
To strictly isolate the impact of weather conditions, the simulation environment was completely cleared of dynamic traffic (other vehicles and pedestrians), and all traffic lights were permanently frozen to green. This ensures that any observed driving failures are purely the result of environmental visual degradation, not unpredictable traffic interactions.

Furthermore, due to the computationally intensive nature of rendering high-fidelity Unreal Engine weather physics while simultaneously running the neural network inference, we were strictly constrained in the total number of possible simulation runs. To maximize statistical validity within these compute and time constraints, we relied on Key Performance Indicators (KPIs).

## Scenario Definition

* **Functional scenario:** A car of the type Lincoln MKZ 2017 is driving in Town02 inside the CARLA simulation, taking route00, with no other dynamic actors on the road and with all stop lights set to green.
* **Logical Scenarios:** Detailed representation of functional scenarios with the help of state space variables. The input Factors are listed below, all drawn from a uniform distribution.

We test five independent environmental Factors:
1. **Sun Altitude** (Glare and lighting angles)
2. **Cloudiness** (Ambient light diffusion)
3. **Precipitation** (Active rain visual noise)
4. **Road Wetness** (Surface reflections and puddles)
5. **Fog Density** (Depth perception and contrast loss)

### Sampling Methodology
The sampling of these parameters is conducted using Latin Hypercube Sampling (LHS) to efficiently cover the parameter space. After iteratively generating and evaluating different sample sizes, we finalized a matrix of 70 runs. This specific size guaranteed that our parameters were sufficiently uniform and orthogonal.

![LHS Sampling Graphs](path/to/your/sampling_graphs.png)

* **Concrete scenarios:** After sampling, we have 70 concrete scenarios.

## Execution
## Execution Architecture

The simulation and testing pipeline is driven by a decoupled architecture, separating the environmental scenario management from the neural network inference. This is handled primarily by two core scripts:

### 1. The Simulation Master (`orchestrator.py`)
This script acts as the high-level environment and scenario manager. Its primary responsibilities include:
* **Matrix Ingestion:** Reading the 70-run Latin Hypercube Sampling (LHS) test matrix.
* **World Initialization:** Connecting to the CARLA server, spawning Town02, and instantiating the ego-vehicle (Lincoln MKZ 2017) at the designated route starting point.
* **Environmental Control:** Dynamically applying the specific continuous weather parameters (Sun Altitude, Road Wetness, Cloudiness, etc.) for each specific run.
* **Data Logging:** Recording the continuous telemetry outputs (Cross-Track Error, Acceleration, Lane Invasions) at a fixed simulation time-step to generate the final analytical datasets.

### 2. The AI Driver (`unified_ai_control.py`)
This script operates as the autonomous agent, completely blind to the "ground truth" of the simulator, relying solely on its sensor suite. Its responsibilities include:
* **Sensor Fusion & Preprocessing:** Capturing the three front-facing RGB camera feeds (Left, Center, Right) and the current ego-speed, applying the necessary normalizations to match the CILv2 training distribution.
* **Inference Engine:** Loading the frozen CILv2 PyTorch model weights and executing the forward pass. It takes the visual feature maps, fuses them with the navigational command token, and calculates the required control vector.
* **Actuation:** Translating the network's continuous outputs back into discrete CARLA control commands (Steer, Throttle, Brake) and applying them to the vehicle chassis for the next simulation frame.

## Analysis

### KPIs
The analysis is centered around four main categories of KPIs:

* **Path adherence:** Measures how well the AI actions adhere to the path defined by the waypoints generated by CARLA. For that, I use Cross Track Error (CTE), which is the perpendicular distance between the current position and the intended reference line. We take the maximum of CTE for each run representing the worst-case scenario.
* **Critical Risk:** Measuring how close the vehicle came to a critical risk, leaving the lane. For that we use TLC which is the estimated time it takes for a vehicle to cross the boundary if it continued on its current trajectory. We take the minimum TLC for each run representing the worst-case scenario.
* **Control confidence:** I use four metrics: maximum and RMS jerk for the lateral and longitudinal movement.
* **Task Failure (Reality):**
  * **Total evasions:** The absolute number of times the vehicle's tires crossed the lane boundary.
  * **Lane Departure Ratio (LDR):** The percentage of the total simulation time the vehicle spent physically outside the bounds of its lane.

### Performance Baseline and Outlier Identification

![KPI Boxplots](analysis/kpi_boxplots.png)

I used the Interquartile Range (IQR) method (+- 1.5 * IQR) to define the operational limits for each metric, and to isolate all the outlier runs for further XAI analysis.

[Insert table of outlier]
