# Glass-Manufacturing-Analyzer
Glass Manufacturing Analyzer - finding relationship between atributtes, handling prediction for multi-input and multi-output solution.

![image](https://github.com/VeRonikARoNik/Glass-Manufacturing-Analyzer/assets/76017554/7fbf5bbc-ced0-49a3-a525-3644997e0093)

![image](https://github.com/VeRonikARoNik/Glass-Manufacturing-Analyzer/assets/76017554/8dd0f8b7-96aa-4c7b-998a-817575dd14ab)
Code 
'''
import pandas as pd
import numpy as np
import sklearn.linear_model as skl
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class GlassManufacturingRegressionAnalyzer:

    def __init__(self, path_to_training_data: str, path_to_validation_data: str, path_to_testing_data: str) -> None:
        """
        Loading and preprocessing of data.
        """
        # 1. Loading training data: input_data => in1, ..., in29 And output_data => out1, ..., out5.
        training_glass_data_frame = pd.read_csv(path_to_training_data)
        training_input_glass_data_frame = pd.DataFrame()
        training_output_glass_data_frame = pd.DataFrame()

        # 1.1 Dividing readed data to input_data (in1, ..., in29) and output_data (out1, ..., out5).
        for column_name in training_glass_data_frame.columns:
            if str(column_name).count("in"):
                training_input_glass_data_frame[column_name]=training_glass_data_frame[column_name]
            elif str(column_name).count("out"):
                training_output_glass_data_frame[column_name]=training_glass_data_frame[column_name]

        # 2. Outliers averaging in input_data (in1, ..., in29).
        for column_name in training_input_glass_data_frame:
            input_data = training_input_glass_data_frame[column_name]
            lb, rb = input_data.quantile(0.05), input_data.quantile(0.95)
            mean = input_data.mean()
            filtred_data = []
            for value in input_data:
                if value > rb or value < lb:
                    filtred_data.append(mean)
                else:
                    filtred_data.append(value)
            training_input_glass_data_frame[column_name] = filtred_data
        
         # 3. Printing basic information.

        print(f"\n============ Loaded Training Data Information ============")
        print(f"First 10 objects: \n{training_input_glass_data_frame.head(10)}")
        print(f"Standard deviation of columns: {training_input_glass_data_frame.std()}")
        print(f"Missing values in the attributes: \n{training_input_glass_data_frame.isna().sum()}")
        print(f"\n==========================================================")

        # 3. Converting training_input_glass_data_frame to training_input_glass_numpy format for handling Machine Learning operation.
        self.training_input_glass_numpy = training_input_glass_data_frame.to_numpy()

        # 4. Converting training_output_glass_data_frame to training_output_glass_numpy format for handling Machine Learning operation.
        self.training_outputs_glass_numpy = {"out1": training_output_glass_data_frame["out1"].to_numpy(), 
                                             "out2": training_output_glass_data_frame["out2"].to_numpy(), 
                                             "out3": training_output_glass_data_frame["out3"].to_numpy(), 
                                             "out4": training_output_glass_data_frame["out4"].to_numpy(),
                                             "out5": training_output_glass_data_frame["out5"].to_numpy()}
        
        # 5. Loading validation data.
        validation_glass_data_frame = pd.read_csv(path_to_validation_data)
        validation_input_glass_data_frame = pd.DataFrame()
        for column_name in validation_glass_data_frame.columns:
            if str(column_name).count("in"):
                validation_input_glass_data_frame[column_name]=validation_glass_data_frame[column_name]
        self.validation_input_glass_numpy = validation_input_glass_data_frame.to_numpy()
        
        # 6. Loading testing data.
        testing_glass_data_frame = pd.read_csv(path_to_testing_data)
        testing_input_glass_data_frame = pd.DataFrame()
        for column_name in testing_glass_data_frame.columns:
            if str(column_name).count("in"):
                testing_input_glass_data_frame[column_name]=testing_glass_data_frame[column_name]
        self.testing_input_glass_numpy = testing_input_glass_data_frame.to_numpy()

    def get_multi_linear_regression_model_for_out(self, out_id:int):
        """
        Creating Linear Regreesion model for output 1.
        """
        if out_id >= 1 and out_id <=5:
            multi_linear_regression_model= skl.LinearRegression()
            input_training_data, output_training_data = self.training_input_glass_numpy, self.training_outputs_glass_numpy[f"out{out_id}"]
            multi_linear_regression_model.fit(input_training_data, output_training_data)
            r_square = multi_linear_regression_model.score(input_training_data, output_training_data)
            print(f"\n========== Multi Linear Regression Model for Out{out_id} =========\n")
            print("a) Validation data: ")
            print(f"Coefficient of determination: {r_square}")
            print(f"Intercept: {multi_linear_regression_model.intercept_}")
            print(f"Cofficients: {multi_linear_regression_model.coef_}")
            print("b) Predicted data:")
            print(f"In1->In29: {self.testing_input_glass_numpy}")
            output_predicted_values = multi_linear_regression_model.predict(self.testing_input_glass_numpy)
            print(f"Out1->Out5: {output_predicted_values}")
            print(f"\n====================================================================")
            return multi_linear_regression_model
    
    # TODO: Random Forest or SVM implementation.

    
    def get_random_forest_model_for_out(self, out_id:int, n_estimators=100):
        """
        Creating Random Forest Regreesion model for output 1.
        """
        if out_id >= 1 and out_id <=5:
            random_forest_model = RandomForestRegressor(n_estimators=n_estimators)
            input_training_data, output_training_data = self.training_input_glass_numpy, self.training_outputs_glass_numpy[f"out{out_id}"]
            random_forest_model.fit(input_training_data, output_training_data)
            r_square = random_forest_model.score(input_training_data, output_training_data)
            print(f"\n========== Random Forest Regression Model for Out{out_id} =========\n")
            print("a) Validation data: ")
            print(f"Coefficient of determination: {r_square}")
            print("b) Predicted data:")
            print(f"In1->In29: {self.testing_input_glass_numpy}")
            output_predicted_values = random_forest_model.predict(self.testing_input_glass_numpy)
            print(f"Out1->Out5: {output_predicted_values}")
            print(f"\n====================================================================")
            return random_forest_model

analyzer = GlassManufacturingRegressionAnalyzer(path_to_training_data="training_input_manufacturing_data.csv", 
                                                path_to_validation_data="validation_input_manufacturing_data.csv",
                                                path_to_testing_data="testing_input_manufacturing_data.csv")

analyzer.get_multi_linear_regression_model_for_out(1)
analyzer.get_multi_linear_regression_model_for_out(2)
analyzer.get_multi_linear_regression_model_for_out(3)
analyzer.get_multi_linear_regression_model_for_out(4)
analyzer.get_multi_linear_regression_model_for_out(5)
analyzer.get_random_forest_model_for_out(1)
analyzer.get_random_forest_model_for_out(2)
analyzer.get_random_forest_model_for_out(3)
analyzer.get_random_forest_model_for_out(4)
analyzer.get_random_forest_model_for_out(5)

'''
![image](https://github.com/VeRonikARoNik/Glass-Manufacturing-Analyzer/assets/76017554/f0cb0768-ae87-46dd-8fa3-dc2abf16fb92)

![image](https://github.com/VeRonikARoNik/Glass-Manufacturing-Analyzer/assets/76017554/7fc53b7e-72da-474b-a8cd-661d0fc76985)

![image](https://github.com/VeRonikARoNik/Glass-Manufacturing-Analyzer/assets/76017554/ca69d7ed-836f-4596-967d-15456d9cf613)
