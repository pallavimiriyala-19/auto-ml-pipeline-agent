import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import logging
import random
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PipelineComponent:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Component({self.name})"

class DataGenerator(PipelineComponent):
    def __init__(self, name="DataGenerator"):
        super().__init__(name)
        self.base_data_mean = 0
        self.base_data_std = 1
        self.drift_factor = 0 # Simulates data drift

    def generate_data(self, num_samples=1000, features=2):
        mean = self.base_data_mean + self.drift_factor
        X = np.random.normal(mean, self.base_data_std, (num_samples, features))
        y = (X[:, 0] + X[:, 1] + np.random.normal(0, 0.5, num_samples) > 0).astype(int)
        logging.debug(f"Generated data with mean: {mean:.2f}")
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(features)]), pd.Series(y, name='target')

    def induce_drift(self, factor=0.5):
        self.drift_factor += factor
        logging.info(f"Data drift induced. New base data mean will be {self.base_data_mean + self.drift_factor:.2f}")

    def reset_drift(self):
        self.drift_factor = 0
        logging.info("Data drift reset.")

class Preprocessor(PipelineComponent):
    def __init__(self, name="Preprocessor"):
        super().__init__(name)
        self.scaler = None # In a real scenario, fit and transform

    def preprocess(self, X):
        # Simple identity preprocessor for this example.
        # In a real scenario, this would apply scaling, encoding, etc.
        logging.debug(f"Preprocessing data. Shape: {X.shape}")
        return X

class ModelTrainer(PipelineComponent):
    def __init__(self, name="ModelTrainer", model_class=LogisticRegression, **model_params):
        super().__init__(name)
        self.model_class = model_class
        self.model_params = model_params if model_params else {'solver': 'liblinear', 'random_state': 42}
        self.model = None

    def train(self, X_train, y_train):
        self.model = self.model_class(**self.model_params)
        self.model.fit(X_train, y_train)
        logging.info(f"Model trained with params: {self.model_params}")
        return self.model

    def evaluate(self, X_test, y_test):
        if not self.model:
            raise ValueError("Model not trained yet.")
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logging.debug(f"Model evaluated. Accuracy: {accuracy:.4f}")
        return {'accuracy': accuracy}

    def update_params(self, new_params):
        self.model_params.update(new_params)
        logging.info(f"Model parameters updated to: {self.model_params}")


class PipelineMetrics:
    def __init__(self, data_stats=None, model_performance=None, resource_usage=None):
        self.data_stats = data_stats if data_stats is not None else {}
        self.model_performance = model_performance if model_performance is not None else {}
        self.resource_usage = resource_usage if resource_usage is not None else {}
        self.timestamp = datetime.now()

    def __repr__(self):
        return f"Metrics(Acc={self.model_performance.get('accuracy', 'N/A'):.2f}, DataMean={self.data_stats.get('feature_0_mean', 'N/A'):.2f})"

class DiagnosisReport:
    def __init__(self, issues=None, severity="LOW"):
        self.issues = issues if issues is not None else []
        self.severity = severity
        self.timestamp = datetime.now()

    def add_issue(self, issue_description, severity="MEDIUM"):
        self.issues.append(issue_description)
        if severity == "HIGH":
            self.severity = "HIGH"
        elif severity == "MEDIUM" and self.severity == "LOW":
            self.severity = "MEDIUM"

    def __bool__(self):
        return bool(self.issues)

    def __repr__(self):
        return f"Diagnosis(Severity={self.severity}, Issues={len(self.issues)})"

class OptimizationStrategy:
    def __init__(self, strategy_type, target_component, params=None):
        self.strategy_type = strategy_type # e.g., 'retrain_model', 'adjust_data_gen'
        self.target_component = target_component # e.g., 'ModelTrainer', 'DataGenerator'
        self.params = params if params is not None else {}
        self.timestamp = datetime.now()

    def __repr__(self):
        return f"Strategy({self.strategy_type} on {self.target_component} with {self.params})"

class MonitorAgent:
    def __init__(self, baseline_accuracy=0.85, baseline_data_mean=0):
        self.baseline_accuracy = baseline_accuracy
        self.baseline_data_mean = baseline_data_mean
        logging.info("Monitor Agent initialized.")

    def collect_metrics(self, data_generator, model_trainer, X_test, y_test):
        data_stats = {
            'feature_0_mean': data_generator.base_data_mean + data_generator.drift_factor,
            'feature_0_std': data_generator.base_data_std
        }
        model_performance = model_trainer.evaluate(X_test, y_test)
        # Simulate resource usage
        resource_usage = {'cpu_percent': random.uniform(10, 50), 'memory_percent': random.uniform(30, 70)}
        return PipelineMetrics(data_stats, model_performance, resource_usage)

class DiagnoseAgent:
    def __init__(self, accuracy_threshold_drop=0.05, data_drift_threshold=0.2):
        self.accuracy_threshold_drop = accuracy_threshold_drop
        self.data_drift_threshold = data_drift_threshold
        self.previous_metrics = None
        logging.info("Diagnose Agent initialized.")

    def diagnose(self, current_metrics, initial_metrics):
        report = DiagnosisReport()

        # Model performance check
        current_accuracy = current_metrics.model_performance.get('accuracy', 0)
        initial_accuracy = initial_metrics.model_performance.get('accuracy', 0)

        if initial_accuracy > 0 and (initial_accuracy - current_accuracy) > self.accuracy_threshold_drop:
            report.add_issue(f"Model accuracy dropped significantly: {initial_accuracy:.2f} -> {current_accuracy:.2f}", "HIGH")

        # Data drift check (simple mean shift)
        current_data_mean = current_metrics.data_stats.get('feature_0_mean', 0)
        initial_data_mean = initial_metrics.data_stats.get('feature_0_mean', 0)
        if abs(current_data_mean - initial_data_mean) > self.data_drift_threshold:
            report.add_issue(f"Data drift detected in feature_0 mean: Initial {initial_data_mean:.2f}, Current {current_data_mean:.2f}", "MEDIUM")

        # Resource usage check (simple threshold)
        if current_metrics.resource_usage.get('cpu_percent', 0) > 80:
             report.add_issue("High CPU usage detected.", "LOW")

        self.previous_metrics = current_metrics
        return report

class StrategizeAgent:
    def __init__(self):
        logging.info("Strategize Agent initialized.")

    def strategize(self, diagnosis_report):
        strategies = []
        if not diagnosis_report:
            return strategies

        for issue in diagnosis_report.issues:
            if "Model accuracy dropped" in issue:
                strategies.append(OptimizationStrategy('retrain_model', 'ModelTrainer', {'solver': 'lbfgs', 'C': random.uniform(0.01, 1.0)}))
                logging.info("Strategizing: Proposing model retraining with new hyperparameters.")
            elif "Data drift detected" in issue:
                strategies.append(OptimizationStrategy('adjust_data_source', 'DataGenerator', {'reset_drift': True}))
                logging.info("Strategizing: Proposing data source adjustment/resetting drift.")
            elif "High CPU usage" in issue:
                strategies.append(OptimizationStrategy('optimize_preprocessor', 'Preprocessor', {'method': 'pca'})) # Placeholder
                logging.info("Strategizing: Proposing preprocessor optimization.")
        return strategies

class ExecuteAgent:
    def __init__(self, pipeline_components):
        self.components = {comp.name: comp for comp in pipeline_components}
        logging.info("Execute Agent initialized.")

    def execute(self, strategy: OptimizationStrategy):
        component = self.components.get(strategy.target_component)
        if not component:
            logging.error(f"Cannot execute strategy: Component '{strategy.target_component}' not found.")
            return False

        if strategy.strategy_type == 'retrain_model' and isinstance(component, ModelTrainer):
            component.update_params(strategy.params)
            # In a real system, you'd trigger a new training run here.
            logging.info(f"Executed: Updated model params for {component.name}.")
            return True
        elif strategy.strategy_type == 'adjust_data_source' and isinstance(component, DataGenerator):
            if strategy.params.get('reset_drift'):
                component.reset_drift()
                logging.info(f"Executed: Reset data drift for {component.name}.")
            return True
        elif strategy.strategy_type == 'optimize_preprocessor' and isinstance(component, Preprocessor):
            logging.info(f"Executed: Applied '{strategy.params.get('method')}' optimization to {component.name}. (Placeholder)")
            return True
        else:
            logging.warning(f"Execution not implemented or invalid for strategy type '{strategy.strategy_type}' on '{component.name}'.")
            return False


class PipelineOptimizer:
    def __init__(self, data_generator, preprocessor, model_trainer):
        self.data_generator = data_generator
        self.preprocessor = preprocessor
        self.model_trainer = model_trainer
        self.pipeline_components = [data_generator, preprocessor, model_trainer]

        self.monitor_agent = MonitorAgent()
        self.diagnose_agent = DiagnoseAgent()
        self.strategize_agent = StrategizeAgent()
        self.execute_agent = ExecuteAgent(self.pipeline_components)

        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.initial_metrics = None
        logging.info("Pipeline Optimizer initialized.")

    def _run_pipeline_cycle(self):
        logging.info("--- Running pipeline cycle ---")
        self.X, self.y = self.data_generator.generate_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        X_train_processed = self.preprocessor.preprocess(self.X_train)
        X_test_processed = self.preprocessor.preprocess(self.X_test)

        self.model_trainer.train(X_train_processed, self.y_train)
        current_metrics = self.monitor_agent.collect_metrics(
            self.data_generator, self.model_trainer, X_test_processed, self.y_test
        )
        logging.info(f"Current Pipeline Metrics: {current_metrics}")
        return current_metrics

    def run_optimization_cycle(self):
        if self.initial_metrics is None:
            logging.info("Performing initial pipeline run to establish baselines...")
            self.initial_metrics = self._run_pipeline_cycle()
            self.diagnose_agent.previous_metrics = self.initial_metrics # Set initial for comparison
            logging.info(f"Baseline Metrics established: {self.initial_metrics}")
            return

        current_metrics = self._run_pipeline_cycle()

        diagnosis = self.diagnose_agent.diagnose(current_metrics, self.initial_metrics)
        if diagnosis:
            logging.warning(f"Diagnosis Report: {diagnosis}")
            strategies = self.strategize_agent.strategize(diagnosis)
            if strategies:
                logging.info(f"Proposed Strategies: {strategies}")
                for strategy in strategies:
                    if self.execute_agent.execute(strategy):
                        logging.info(f"Strategy '{strategy.strategy_type}' applied successfully.")
                        # After applying a strategy, we might want to re-evaluate or retrain
                        # For simplicity, next cycle will pick up the changes.
                        # Re-run pipeline to see effect immediately if desired
                        self._run_pipeline_cycle() # Re-run after an execution to see immediate effect
                    else:
                        logging.error(f"Failed to execute strategy: {strategy}")
            else:
                logging.info("No concrete strategies proposed for the diagnosed issues.")
        else:
            logging.info("No significant issues detected. Pipeline operating normally.")