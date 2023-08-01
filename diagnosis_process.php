<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $patient_name = $_POST["patient_name"];
    $patient_age = $_POST["patient_age"];
    $cancer_type = $_POST["cancer_type"];

    // Assuming you have Python installed and added to your system PATH
    // Replace 'python3' with 'python' if you are on Windows
    $python_executable = 'python';
    $script_path = 'C:\Users\Loges\PycharmProjects\pythonProject5\Cancer_prediction_1\cancer_diagnosis.py'; // Replace with the actual path to your Python script

    // Construct the command to call the Python script
    $command = "$python_executable $'C:\Users\Loges\PycharmProjects\pythonProject5\Cancer_prediction_1\cancer_diagnosis.py' \"$cancer_type\"";

    // Execute the command and get the output
    $output = shell_exec($command);

    // Print the diagnosis result
    echo "Patient Name: $patient_name<br>";
    echo "Patient Age: $patient_age<br>";
    echo "Cancer Type: $cancer_type<br>";
    echo "Diagnosis Result: $output";
}
?>
