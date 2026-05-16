# FUNCTIONALITY.md: AutoML Pipeline Agent

## Overview

The AutoML Pipeline Agent is an autonomous multi-agent system designed to monitor, diagnose, and optimize machine learning pipelines in real-time. It aims to reduce manual MLOps burden, improve model reliability, and ensure continuous high performance by proactively addressing issues like data drift, model decay, and resource bottlenecks.

## Core Architecture

The system is composed of several specialized agents that operate in a continuous feedback loop, orchestrated by a central `PipelineOptimizer`.

### 1. Pipeline Components

These are the modular building blocks of a typical ML pipeline that the agents interact with:

*   **`DataGenerator`**: Simulates data ingestion, potentially introducing data drift over time. In a real-world scenario, this would interface with actual data sources (databases, streaming platforms).
*   **`Preprocessor`**: Simulates data transformation steps. It can be optimized based on agent recommendations.
*   **`ModelTrainer`**: Responsible for training and evaluating an ML model. It can update its internal model parameters based on optimization strategies.

### 2. Agent Roles

#### a. `MonitorAgent`

*   **Role**: Observes the current state and performance of the ML pipeline.
*   **Data Inputs**: Receives access to the `DataGenerator` (for data statistics) and `ModelTrainer` (for model performance metrics) along with test data.
*   **Outputs**: Produces a `PipelineMetrics` object, which encapsulates:
    *   `data_stats`: Descriptive statistics of the input data (e.g., mean, standard deviation of features).
    *   `model_performance`: Evaluation metrics of the trained model (e.g., accuracy, F1-score, loss).
    *   `resource_usage`: Simulated or actual resource consumption (e.g., CPU, memory).

#### b. `DiagnoseAgent`

*   **Role**: Analyzes the `PipelineMetrics` to identify deviations from expected behavior and pinpoint potential issues.
*   **Data Inputs**: `PipelineMetrics` (current and initial/baseline metrics).
*   **Logic**: Compares current metrics against established baselines or historical trends. It employs rule-based heuristics to detect common MLOps problems:
    *   **Model Performance Degradation**: If model accuracy drops below a predefined threshold relative to the baseline.
    *   **Data Drift**: If key statistical properties of the input data (e.g., feature means) significantly shift from the baseline.
    *   **Resource Bottlenecks**: If simulated resource usage exceeds acceptable limits.
*   **Outputs**: Generates a `DiagnosisReport` which lists identified `issues` (e.g., "Model accuracy dropped," "Data drift detected") and assigns an overall `severity` (LOW, MEDIUM, HIGH).

#### c. `StrategizeAgent`

*   **Role**: Based on the `DiagnosisReport`, it proposes concrete optimization strategies.
*   **Data Inputs**: `DiagnosisReport`.
*   **Logic**: Maps diagnosed issues to predefined or dynamically generated optimization actions. In this implementation, it uses simple `if/elif` rules:
    *   **Model Accuracy Drop**: Suggests `retrain_model` with potentially new hyperparameters (e.g., different regularization strength `C`).
    *   **Data Drift**: Suggests `adjust_data_source`, which could involve recalibrating data generation or resetting drift.
    *   **Resource Usage**: Suggests `optimize_preprocessor` (e.g., using PCA), which is a placeholder for more complex resource-optimization strategies.
*   **Outputs**: A list of `OptimizationStrategy` objects, each specifying a `strategy_type`, `target_component`, and `params` (e.g., `{'C': 0.5}` for a model).

#### d. `ExecuteAgent`

*   **Role**: Implements the proposed `OptimizationStrategy` on the target pipeline component.
*   **Data Inputs**: `OptimizationStrategy` and references to the actual `PipelineComponent` objects.
*   **Logic**: Contains methods to interact with `DataGenerator`, `Preprocessor`, and `ModelTrainer` to apply the changes specified in the strategy. This could involve updating model parameters, triggering data reloading, or reconfiguring preprocessing steps.
*   **Outputs**: Reports the success or failure of the execution.

### 3. `PipelineOptimizer` (Orchestrator)

*   **Role**: The central coordinator that manages the flow between the pipeline components and the agents.
*   **Workflow (`run_optimization_cycle` method)**:
    1.  **Initial Baseline**: On the first run, it executes the pipeline to establish `initial_metrics` as a baseline for future comparisons.
    2.  **Regular Cycle**: In subsequent runs:
        *   It triggers the `DataGenerator` to produce new data.
        *   Data is passed through the `Preprocessor`.
        *   The `ModelTrainer` trains and evaluates the model.
        *   The `MonitorAgent` collects `current_metrics`.
        *   The `DiagnoseAgent` analyzes `current_metrics` against `initial_metrics`.
        *   If issues are found, the `StrategizeAgent` proposes `OptimizationStrategy` instances.
        *   The `ExecuteAgent` attempts to apply these strategies to the relevant `PipelineComponent`s.
        *   After execution, the pipeline typically runs again to observe the immediate impact of the changes.

## Data Flow

1.  **Pipeline Components** (`DataGenerator`, `Preprocessor`, `ModelTrainer`) produce data and metrics.
2.  **`MonitorAgent`** collects `PipelineMetrics` from the components.
3.  **`DiagnoseAgent`** receives `PipelineMetrics` and `initial_metrics` and generates a `DiagnosisReport`.
4.  **`StrategizeAgent`** receives `DiagnosisReport` and creates `OptimizationStrategy` objects.
5.  **`ExecuteAgent`** receives `OptimizationStrategy` objects and applies changes back to the **Pipeline Components**.
6.  The **`PipelineOptimizer`** orchestrates this continuous loop.

## Design Decisions

*   **Modularity**: Each agent and pipeline component is a distinct class, promoting reusability and extensibility. New monitoring checks, diagnostic rules, or optimization strategies can be added with minimal impact on existing code.
*   **Simplicity for Demonstration**: For clarity, many real-world complexities (e.g., asynchronous operations, complex LLM prompts, persistent state management, robust error handling, actual cloud resource monitoring) are simplified or abstracted. The `DataGenerator` simulates data drift, and `resource_usage` is random.
*   **Rule-Based Agents**: The `DiagnoseAgent` and `StrategizeAgent` use basic rule-based logic. This can be easily extended with more sophisticated ML models, reinforcement learning agents, or integrated with LLMs for advanced reasoning capabilities.
*   **Feedback Loop**: The explicit feedback loop (Execute Agent applies changes, next cycle observes effects) is crucial for an adaptive system.
*   **Clear Data Structures**: Dedicated classes like `PipelineMetrics`, `DiagnosisReport`, and `OptimizationStrategy` ensure clear communication between agents.

This architecture provides a robust foundation for building truly autonomous and self-optimizing ML pipelines.