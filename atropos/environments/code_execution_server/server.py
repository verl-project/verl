"""
Instructions:

# Build the image
docker build -t cpp-flask-executor .

# Run the container
docker run -p 5002:5002 cpp-flask-executor

curl -X POST http://localhost:5002/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "#include<iostream>\nint main(){int x; std::cin>>x; std::cout<<x*x; return 0;}", "input": "4"}'



"""

import os
import subprocess
import uuid

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/execute", methods=["POST"])
def execute_code():
    try:
        # Receive C++ code from API request
        data = request.json
        code = data.get("code", "")
        test_cases = data.get("input", "")

        if not code:
            return jsonify({"error": "No code provided"}), 400

        file_id = str(uuid.uuid4())
        py_filename = f"/tmp/{file_id}.py"

        # Save the Python code to a file
        with open(py_filename, "w") as f:
            f.write(code)

        # Execute Python code using subprocess
        exec_result = subprocess.run(
            ["python", py_filename],
            input=test_cases,
            capture_output=True,
            text=True,
            timeout=5,  # Prevent infinite loops
        )

        # Cleanup temporary files
        os.remove(py_filename)

        return jsonify({"output": exec_result.stdout, "error": exec_result.stderr})
        """ C++ stuff

        file_id = str(uuid.uuid4())
        cpp_filename = f"/tmp/{file_id}.cpp"
        exe_filename = f"/tmp/{file_id}.out"

        # Save code to a file
        with open(cpp_filename, "w") as f:
            f.write(code)

        # Compile C++ code
        compile_result = subprocess.run(["g++", cpp_filename, "-o", exe_filename],
                                        capture_output=True, text=True)

        for i in range(5):
            print(i)
            time.sleep(1)

        if compile_result.returncode != 0:
            return jsonify({"error": "Compilation failed", "details": compile_result.stderr})

        print(test_cases)
        # Execute compiled binary
        exec_result = subprocess.run([exe_filename], input=test_cases, capture_output=True, text=True, timeout=5)

        # Cleanup temporary files
        os.remove(cpp_filename)
        os.remove(exe_filename)

        return jsonify({"output": exec_result.stdout, "error": exec_result.stderr})"""

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
