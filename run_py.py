from ecologits import EcoLogits
from openai import OpenAI
import os
import subprocess
import sys
from enum import Enum

class Result(Enum):
    SUCCESS = 0
    COMPILATION_FAILURE = 1
    RUNTIME_FAILURE = 2
    RESULT_MISMATCH = 3
    TIMEOUT = 4


def run_with_stdin(process_name, input_data):
    try:
        result = subprocess.run(
            process_name,
            input=input_data,           # Pass input as bytes
            stdout=subprocess.PIPE,     # Capture standard output
            stderr=subprocess.PIPE,     # Capture standard error
            timeout=100,
            text=True                   # Return output as string
        )
    except subprocess.TimeoutExpired as e:
        print(f"Command timed out: {e}")
        return Result.TIMEOUT, ""
    else:
        return result.returncode, result.stdout + "\n" + result.stderr

def eliminate_tag(sourceCode):
    lines = sourceCode.split("\n")
    start = -1
    end = -1
    for i in range(0, len(lines)):
        if "```python" in lines[i]:
            start = i
        elif start >= 0 and "```" in lines[i]:
            end = i
            break

    if start < 0:
        return sourceCode
    else:
        return "\n".join(lines[start+1:end])

def count_files_by_extension_in_directory(directory, extension):
  count = 0
  for file in os.listdir(directory):
    if file.endswith(extension):
      count += 1
  return count

def read_input_output(folderName):
    test_count = count_files_by_extension_in_directory(folderName, "in")
    inputs = []
    expected_outputs = []
    for t in range(test_count):
        inputName = folderName + "/" + str(t+1) + ".in"
        outputName = folderName + "/" + str(t+1) + ".out"

        f = open(inputName, 'r')
        inputs.append(f.read())
        f.close()

        f = open(outputName, 'r')
        expected_outputs.append(f.read())
        f.close()
    return (inputs, expected_outputs)

def init_chatgpt(api_key):
    chatgpt = OpenAI(api_key=api_key)
    return chatgpt

def send_chatgpt_prompt(messages, client, trial, model):
    print("#####################################################")
    print("Trial: " + str(trial))
    print("#####################################################")
    print(messages)
    print("-----------------------------------------------------")
    """
        gpt-4o-mini (0.15/0.6)
        gpt-4o: default (5/15)
        gpt-4-turbo (10/30)
        gpt-4 (30/60)

        gpt-3.5-turbo-1106 (1/2)
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
    )

    # Get estimated environmental impacts of the inference
    print("-----------------------------------------------------")
    print("# of input tokens: " + str(response.usage.prompt_tokens))
    print("# of output tokens: " + str(response.usage.completion_tokens))
    print("-----------------------------------------------------")
    for choice in response.choices :
        print(choice.message.content)
    print("-----------------------------------------------------")
    print(f"Energy consumption: {response.impacts.energy.value} kWh")
    print(f"GHG emissions: {response.impacts.gwp.value} kgCO2eq")

    return response.choices[0].message.content.strip()

def check_response(response, folderName, inputs, expected_outputs):

    sourceCodeName = folderName + "/source.py"
    f = open(sourceCodeName, 'w')
    sourceCode = response
    sourceCode = eliminate_tag(sourceCode)
    f.write(sourceCode)
    f.close()

    test_count = count_files_by_extension_in_directory(folderName, "in")

    for t in range(test_count):

        inputName = folderName + "/" + str(t+1) + ".in"
        
        print("-----------------------------------------------------")
        print("Testing: " + str(inputName))
        print("-----------------------------------------------------")

        exit_code, output = run_with_stdin(["python3", sourceCodeName], inputs[t])
        if exit_code == Result.TIMEOUT:
            return (Result.TIMEOUT, "")
        elif exit_code == 0:
            if output.rstrip() == expected_outputs[t].rstrip():
                print("Execution and expectation matches")
            else:
                print("Execution and expectation differs")
                print("--- Execution ---")
                print(output.rstrip())
                print("--- Expectation ---")
                print(expected_outputs[t].rstrip())
                return (Result.RESULT_MISMATCH, output, t)
        else:
            print("Execution failed:")
            print(output)
            return (Result.RUNTIME_FAILURE, output)
            
    return (Result.SUCCESS, "")


# Initialize EcoLogits
EcoLogits.init()

folder = sys.argv[1]
model = sys.argv[2]
if model == "":
    model = "gpt-4o"

chatgpt = init_chatgpt(sys.argv[3])

conversation_history = [
        {"role": "system", "content": "You are a world-class programmer for computer science olympiad competition."}
]

f = open(folder + "/prob.txt", 'r')
problem = f.read()
f.close()

inputs, expected_outputs = read_input_output(folder)

prompt = 'JUST EMIT PYTHON CODE for the following Olympiad-level computer science problem: ' + problem + '\n'

conversation_history.append({"role": "user", "content": prompt})

print("########################################")
print("Model: " + model)
print("########################################")

num_trials = 0
while num_trials < 25:
    response = send_chatgpt_prompt(conversation_history, chatgpt, num_trials, model)
    if num_trials > 10:
        conversation_history.pop(2)
        conversation_history.pop(2)

    conversation_history.append({"role": "assistant", "content": response})
    result = check_response(response, folder, inputs, expected_outputs)
    num_trials += 1
    if result[0] == Result.COMPILATION_FAILURE:
        prompt = "Your code failed in compilation. The first 3000 characeters of the log is as follows: " + result[1][:3000]
    elif result[0] == Result.TIMEOUT:
        prompt = "Your code took long and failed to finish within 100 seconds."
    elif result[0] == Result.RUNTIME_FAILURE:
        prompt = "Your code failed at runtime. The first 3000 characeters of the log is as follows: " + result[1][:3000]
    elif result[0] == Result.RESULT_MISMATCH:
        test_num = result[2]
        output = result[1]
        prompt = "Result mismatched. The input was as follows:\n" + inputs[test_num] + "The expected output was as follows:\n" + expected_outputs[test_num] + "However, the actual output was as follows:\n" + output + "\n"
        if (test_num > 0):
            prompt = prompt + "On the other hand, your code produced correct output for the following cases:\n"
            for t in range(test_num):
                prompt += "Input: " + inputs[t] + "\nOutput: " + expected_outputs[t] + "\n"
    else:
        break

    prompt = prompt + ". Can you review your code thoroughly and fix the code?\n"

    conversation_history.append({"role": "user", "content": prompt})
