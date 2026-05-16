import time
from auto_ml_pipeline_agent.main_code import DataGenerator, Preprocessor, ModelTrainer, PipelineOptimizer

# --- Example Usage for AutoML Pipeline Agent ---

# 1. Initialize your individual ML pipeline components
#    These components simulate parts of a real ML pipeline.
#    DataGenerator: Creates synthetic data, can simulate drift.
#    Preprocessor: Simple placeholder for data transformation.
#    ModelTrainer: Trains and evaluates a Logistic Regression model.

data_gen = DataGenerator()
preprocessor = Preprocessor()
model_trainer = ModelTrainer(model_class=LogisticRegression, solver='liblinear', random_state=42)

# 2. Instantiate the PipelineOptimizer
#    This orchestrator holds the agents and the pipeline components.
optimizer = PipelineOptimizer(data_gen, preprocessor, model_trainer)

print("====================================================")
print("     Starting AutoML Pipeline Agent Demonstration     ")
print("====================================================")

# --- Step 1: Establish Baseline ---
# The first run initializes the pipeline and gathers baseline metrics.
# The agents will use these baselines for comparison in future cycles.
print("\n--- Running Initial Cycle (Establishing Baseline) --- ")
optimizer.run_optimization_cycle()
print("Initial pipeline baseline established. No issues expected.\n")

# --- Step 2: Run Healthy Cycles ---
# Observe the pipeline operating normally. No major issues should be detected.
print("--- Running 2 Healthy Pipeline Cycles --- ")
for i in range(1, 3):
    print(f"\n>>> Cycle {i+1}: Healthy Operation <<< ")
    optimizer.run_optimization_cycle()
    time.sleep(1) # Simulate time passing between cycles
print("Pipeline operating normally. No actions taken.\n")

# --- Step 3: Induce Data Drift ---
# We'll now simulate a real-world scenario where the input data distribution changes.
# This should trigger the DiagnoseAgent.
print("--- Inducing Significant Data Drift in DataGenerator --- ")
data_gen.induce_drift(factor=0.6)
print("Data drift introduced. Expecting agent intervention soon.\n")

# --- Step 4: Run Cycles with Drift (Expect Diagnosis and Optimization) ---
# The agents should detect the drift, diagnose the problem, propose a strategy,
# and execute it (in this case, resetting the drift and suggesting model retraining).
print("---> Running 3 Cycles with Data Drift <--- ")
for i in range(3, 6):
    print(f"\n>>> Cycle {i+1}: With Data Drift <<< ")
    optimizer.run_optimization_cycle()
    time.sleep(1)
print("Agents should have detected drift and initiated corrective actions.\n")

# --- Step 5: Observe Post-Correction Performance ---
# After the agents have taken action (like resetting drift and suggesting retraining),
# we should see the pipeline's performance stabilize or improve.
print("--- Running 2 Post-Correction Cycles --- ")
for i in range(6, 8):
    print(f"\n>>> Cycle {i+1}: Post-Correction <<< ")
    optimizer.run_optimization_cycle()
    time.sleep(1)
print("Pipeline should now be recovering or stable after agent interventions.\n")

print("====================================================")
print("   AutoML Pipeline Agent Demonstration Completed!   ")
print("====================================================")
print("\nReview the logs above to see agent actions and pipeline state changes.")
