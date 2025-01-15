"""
"""
import importlib
import zipfile
import shutil
import os
import sys
import subprocess
import datetime
import argparse
try:
    import requests
except:
    pass
from timeit import default_timer as timer
from argparse import RawTextHelpFormatter

SCRIPT_ABS_DIR = os.path.abspath(__file__).replace("reproduce_paper.py", "")

PROJECT_NAME = "IncDISCES"
VERSION = "1.0.0"
# ZIP_LOCATION = "https://github.com/rebesatt/disces/blob/main/datasets/DISCES.zip"
# FINANCE_ZIP_LOCATION = "https://github.com/rebesatt/disces/tree/main/datasets/finance/finance.zip"
# GOOGLE_ZIP_LOCATION = "https://github.com/rebesatt/disces/blob/main/datasets/google/google.zip"

PYTHON_COMMAND = "python"

def print_to_std_and_file(str_to_log):
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(current_timestamp + ": " + str_to_log)
    with open(log_file, 'a') as f:
        f.write(current_timestamp + ": " + str_to_log + "\n")

def print_with_timestamp(str_to_stdout):
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(current_timestamp + ": " + str_to_stdout)

def verify_requirements():
    try:
        if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
            print_to_std_and_file("Running this script requries Python >= 3.10. Current versions is " + str(sys.version_info.major) + "." + str(sys.version_info_minor))
            sys.exit(1)
    except:
        print_to_std_and_file("Running this script requries Python >= 3.10. Please upgrade your Python and retry again.")
        sys.exit(1)
    any_package_missing = False
    try:
        import requests
    except:
        print_to_std_and_file("Python package 'requests' is not installed.")
        any_package_missing = True
    try:
        import pandas
    except:
        print_to_std_and_file("Python package 'pandas' is not installed.")
        any_package_missing = True
    # try:
    #     import msgpack
    # except:
    #     print_to_std_and_file("Python package 'msgpack' is not installed.")
    #     any_package_missing = True
    try:
        import numpy
    except:
        print_to_std_and_file("Python package 'numpy' is not installed.")
        any_package_missing = True
    try:
        import matplotlib
    except:
        print_to_std_and_file("Python package 'matplotlib' is not installed.")
        any_package_missing = True
    try:
        import seaborn
    except:
        print_to_std_and_file("Python package 'seaborn' is not installed.")
        any_package_missing = True
    # try:
    #     import func_timeout
    # except:
    #     print_to_std_and_file("Python package 'func_timeout' is not installed.")
    #     any_package_missing = True
    try:
        import jinja2
    except:
        print_to_std_and_file("Python package 'jinja2' is not installed.")
        any_package_missing = True
    try:
        import typing_extensions
    except:
        print_to_std_and_file("Python package 'typing_extensions' is not installed.")
        any_package_missing = True

    if any_package_missing:
        print_to_std_and_file("This script requries all mentioned packages. Please install them manually.")
        print_to_std_and_file("The missing packages can be installed in 'DISCES/' via '" + PYTHON_COMMAND + " -m pip install -r requirements.txt'")
        if not shutil.which("latex"):
            print_to_std_and_file("In addtion reproducing this script requries LaTeX. No LaTeX installation found.")
        sys.exit(1)
    else:
        if not shutil.which("latex"):
            print_to_std_and_file("Note: Reproducing this script requries LaTeX. No LaTeX installation found.")

def execute_shell_command(cmd, stdout_verification=None, b_stderr_verification=False):
    print_to_std_and_file("Executed command: " + cmd)
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_str = str(result.stdout)[2:-1]
    stderr_str = str(result.stderr)[2:-1]
    if len(stderr_str):
        if len(stdout_str):
            print_to_std_and_file("STDOUT:")
            splitted_output = stdout_str.replace("\\r", '\r').replace("\\n", '\n').split('\n')
            for line in splitted_output:
                print_to_std_and_file(line)
        print_to_std_and_file("STDERR:")
        splitted_output = stderr_str.replace("\\r", '\r').replace("\\n", '\n').split('\n')
        for line in splitted_output:
            print_to_std_and_file(line)
    else:
        splitted_output = stdout_str.replace("\\r", '\r').replace("\\n", '\n').split('\n')
        for line in splitted_output:
            print_to_std_and_file(line)

    if stdout_verification is not None:
        if stdout_verification not in result.stdout.decode():
            print_to_std_and_file("Verification string " + stdout_verification + " not in stdout")
            raise Exception
    if b_stderr_verification:
        if result.stderr != "b''":
            print_to_std_and_file("stderr is not empty: " + str(result.stderr[2:-1]))
            raise Exception

def execute_shell_command_with_live_output(cmd):
    print_to_std_and_file("Executed command: " + cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print_to_std_and_file(output.strip().decode('utf-8'))
    rc = process.poll()
    if rc != 0:
        print_to_std_and_file("RC not 0 (RC=" + str(rc) + ") for command: " + cmd)
        raise Exception
    return rc

# def download_repository():
#     if os.path.isdir(SCRIPT_ABS_DIR + "datasets/") and os.path.isdir(SCRIPT_ABS_DIR + "docs/") and os.path.isdir(SCRIPT_ABS_DIR + "experiments/"):
#         print_to_std_and_file("The reproduce_paper.py script is part of a '" + PROJECT_NAME + "' project. No files will be downloaded.")
#         project_root = SCRIPT_ABS_DIR
#         return project_root

#     print_to_std_and_file("The reproduce_paper.py script is a standalone. Downloading '" + PROJECT_NAME + "'s source code.")
#     zipped_project = SCRIPT_ABS_DIR + "/" + PROJECT_NAME + ".zip"
#     project_root = SCRIPT_ABS_DIR + PROJECT_NAME + "/"

#     zipped_project = 'IncDISCES.zip'
#     print_to_std_and_file("Downloading '" + PROJECT_NAME + "'s source code ...")
#     # response = requests.get(ZIP_LOCATION)
#     if response.status_code == 200:
#         with open(zipped_project, mode='wb') as file:
#             file.write(response.content)
#         print_to_std_and_file("Successfully downloaded '" + PROJECT_NAME + "'s source code to " + zipped_project + ".")

#         print_to_std_and_file("Extract " + zipped_project + ".")
#         try:
#             with zipfile.ZipFile(zipped_project, 'r') as zip_ref:
#                 zip_ref.extractall(".temp_" + PROJECT_NAME)
#             os.remove(zipped_project)
#             print_to_std_and_file("All files extracted and " + zipped_project + " deleted.")
#         except Exception as e:
#             print_to_std_and_file(e.message)
#             sys.exit(1)
#     else:
#         print_to_std_and_file("Download failed. Status code " + str(response.status_code))
#         sys.exit(1)

#     if os.path.isdir(project_root):
#         shutil.copytree(".temp_" + PROJECT_NAME + "/" + PROJECT_NAME, PROJECT_NAME, dirs_exist_ok=True)
#         print_to_std_and_file("Found existing local project " + project_root + ". Merge repository updates to it.")
#     else:
#         shutil.move(".temp_" + PROJECT_NAME + "/" + PROJECT_NAME, PROJECT_NAME)
#         print_to_std_and_file("Downloaded the project to " + project_root)
#     shutil.rmtree(".temp_" + PROJECT_NAME)


#     shutil.copyfile(SCRIPT_ABS_DIR + "reproduce_paper.py", SCRIPT_ABS_DIR + "DISCES/reproduce_paper.py")
#     os.chdir(project_root)
#     # download_and_extract_datasets("finance", FINANCE_ZIP_LOCATION)
#     # download_and_extract_datasets("google", GOOGLE_ZIP_LOCATION)
#     # os.chdir(SCRIPT_ABS_DIR)
#     # print_to_std_and_file("To prevent this script from downloading on each run, use the 'reproduce_paper.py' script in '" + SCRIPT_ABS_DIR + "DISCES/'")

#     return project_root

# def download_and_extract_datasets(name, location):
#     os.chdir("datasets/" + name + "/")
#     zipped_data = name + '.zip'
#     print_to_std_and_file("Downloading '" + zipped_data + "' ...")
#     response = requests.get(location)
#     if response.status_code == 200:
#         with open(zipped_data, 'wb') as file:
#             file.write(response.content)
#         print_to_std_and_file("Successfully downloaded " + name + " datasets to 'datasets/" + name + "/" + zipped_data + "'.")

#         print_to_std_and_file("Extract " + zipped_data + ".")
#         with zipfile.ZipFile(zipped_data, 'r') as zip_ref:
#             zip_ref.extractall(".temp_" + name)
#         shutil.copytree(".temp_" + name + "/", "../" + name, dirs_exist_ok=True)
#         os.remove(zipped_data)
#         shutil.rmtree(".temp_" + name)
#         print_to_std_and_file("All files extracted and " + zipped_data + " deleted.")
#     else:
#         print_to_std_and_file("Download of " + zipped_data + " failed. Status code " + str(response.status_code))

#     os.chdir("../../")

# def create_synthetic_data():
#     cur_wd = os.getcwd()
#     os.chdir("datasets/synt/")
#     execute_shell_command_with_live_output(PYTHON_COMMAND + " synt_sample_generator.py")
#     os.chdir(cur_wd)

def create_verification_file(local_result_dir, file_to_verify_execution):
    f = open(local_result_dir + "/" + file_to_verify_execution, "w")
    f.write("Done")
    f.close()

def run_experiments(local_result_dir, run_command_prefix, functions, test_name_prefix, file_to_verify_execution, args=None, estimated_runtimes=None):
    # if os.path.isfile(local_result_dir + "/" + file_to_verify_execution):
    #     print_to_std_and_file("Found existing test folder. Skipping.")
    # else:
    b_create_verification_file = True
    for i, function in enumerate(functions):
        cmd = str(run_command_prefix) + " " + str(test_name_prefix) + str(function) + ".py"
        if args:
            cmd += " " + str(args[i])
        if estimated_runtimes:
            print_to_std_and_file("Experiment " + str(test_name_prefix) + str(function) + " estimated runtime is " + str(estimated_runtimes[i]) + ".")
        try:
            start = timer()
            execute_shell_command_with_live_output(cmd)
            end = timer()
            print_to_std_and_file("Experiment " + str(test_name_prefix) + str(function) + " took " + str(end - start) + " seconds")
        except Exception as msg:
            end = timer()
            print_to_std_and_file(str(msg))
            print_to_std_and_file("Experiment " + str(test_name_prefix) + str(function) + " failed after " + str(end - start) + " seconds")
            b_create_verification_file = False
    if b_create_verification_file:
        create_verification_file(local_result_dir, file_to_verify_execution)
    assert os.path.isfile(local_result_dir + "/" + file_to_verify_execution)

def compile_main_pdf():
    figure_list = {"Figure 1": "support_distribution_all.pdf",
                   "Figure 2": "gen_sample.pdf",
                   "Figure 3": "gen_supp.pdf",
                   "Figure 4": "gen_sampsupp.pdf",
                   "Figure 5": "spec_sample.pdf",
                   "Figure 6": "spec_supp.pdf",
                   "Figure 7": "spec_sampsupp.pdf",
                   }
    # tex_list = {"Table 1": "sota_acc.tex",
    #             "Table 2": "char_corr.tex",
    #             "Table 3": "rl_compare.tex"}
    if not os.path.isdir(project_root + "docs/img"):
        print_to_std_and_file("'" + project_root + "docs//img' was not available. All original images are missing.")
        os.makedirs(project_root + "docs/img")
        print_to_std_and_file("'" + project_root + "docs/img' created.")
    for file_list in [figure_list]:
        if file_list == figure_list:
            img_path = "img/"
            file_type = "plot"

        for file_name, file in file_list.items():
            new_file_exists = os.path.isfile(local_result_dir + "/" + file)
            cur_file_exists = os.path.isfile(project_root + "docs/" + img_path + file)
            old_file_exists = os.path.isfile(project_root + "docs/originals/" + file)
            if new_file_exists:
                if cur_file_exists and not old_file_exists:
                    if not os.path.isdir(project_root + "docs/originals"):
                        os.makedirs(project_root + "docs/originals")
                        print_to_std_and_file("'" + project_root + "docs/originals' created.")
                    print_to_std_and_file("Replacing " + file_name + " (" + file + ") with the reproduced one. Original file can be found in '" + project_root + "docs/originals/'")
                    shutil.copyfile(project_root + "docs/" + img_path + file, project_root + "docs/originals/" + file)
                elif old_file_exists:
                    print_to_std_and_file("Copying " + file_name + " (" + file + ") to '" + project_root + "docs/" + img_path + "'")
                    print_to_std_and_file("Note: original " + file_name + " (" + file + ") is in '" + project_root + "docs/originals/' available.")
                else:
                    print_to_std_and_file("Copying " + file_name + " (" + file + ") to '" + project_root + "docs/" + img_path + "'")
                    print_to_std_and_file("Note: original " + file_name + " (" + file + ") is not available.")
                shutil.copyfile(local_result_dir + "/" + file, project_root + "docs/" + img_path + file)
            else:
                if cur_file_exists and not old_file_exists:
                    print_to_std_and_file("Note: " + file_name + " (" + file + ") wasn't found in '" + local_result_dir + "'. Using the original " + file_type + " from the paper.")
                if not cur_file_exists:
                    if old_file_exists:
                        print_to_std_and_file("Moving " + file_name + " (" + file + ") to '" + project_root + "docs/" + img_path + "'")
                        print_to_std_and_file("Note: " + file_name + " (" + file + ") wasn't found in '" + local_result_dir + "'. Using the original " + file_type + " from the paper.")
                        shutil.move(project_root + "docs/originals/" + file, project_root + "docs/" + img_path +file)
                    else:
                        print_to_std_and_file("Note: " + file_name + " (" + file + ") wasn't found in '" + project_root + "docs/originals/'." +
                        "The " + file_type + " will not be included in the paper.")

    os.chdir(project_root + "docs/")

    execute_shell_command('pdflatex --interaction=nonstopmode main.tex')
    execute_shell_command('bibtex main.aux')
    execute_shell_command('pdflatex --interaction=nonstopmode main.tex')
    # try:
    #     execute_shell_command('pdflatex --interaction=nonstopmode main.tex', stdout_verification="main.pdf")
    # except Exception:
    if not os.path.exists('main.pdf'):
        print_to_std_and_file("The paper, containing the new figures based on these experiments, could not be generated. Check the log for more details.")
        sys.exit(1)

    shutil.copyfile("main.pdf", reproduced_main_pfd_file)
    print_to_std_and_file("The reproduced paper, containing the new figures based on these experiments, is in " + reproduced_main_pfd_file)

def clean_working_directory():
    try:
        os.remove("reproduce_paper.log")
    except:
        pass
    shutil.rmtree("IncDISCES/", ignore_errors=True)
    # shutil.rmtree("datasets/synt/domain_size/", ignore_errors=True)
    # shutil.rmtree("datasets/synt/max_pattern/", ignore_errors=True)
    # shutil.rmtree("datasets/synt/max_type/", ignore_errors=True)
    # shutil.rmtree("datasets/synt/trace_length/", ignore_errors=True)
    # shutil.rmtree("datasets/synt/traces/", ignore_errors=True)
    # shutil.rmtree("datasets/synt/types/", ignore_errors=True)
    shutil.rmtree("experiments/results/", ignore_errors=True)
    try:
        os.chdir(project_root + "docs/")
        execute_shell_command('pdflatex -C')
    except:
        pass

if __name__ == "__main__":
    # Setup
    parser = argparse.ArgumentParser(description =  "This script reproduces the results of the " + PROJECT_NAME + " project. It checks required libaries, downloads necessary dataset and scripts, runs the experiments, and compiles the paper with the new results if a latex compiler is available.\n"
            "Requirements:\n"
            "(1) Python >= 3.10",
            formatter_class=RawTextHelpFormatter)
    parser.add_argument("-dd", dest="b_download_dataset",
            help="if the argument -dd is given, the script only downloads the repository and external datasets and exits (without running any experiments).",
            action='store_true')
    # parser.add_argument("-ilm", dest="b_run_extended_il_miner",
    #         help="if the argument -ilm is given, the experiments will include runs of the il-miner, which needs approximately additional 7 days.",
    #         action="store_true")
    parser.add_argument("-noexp", dest="b_run_no_experiment",
            help="if the argument -noexp is given, the script skips the experiments and only (if necessary) downloads the project and complies the current version of the paper.",
            action="store_true")
    parser.add_argument("-pv", dest="s_python_version",
            nargs='?',
            help="-pv lets you specify another command to run all python scripts, e.g. using 'pyhton3' instead of 'python' (default).",
            type=str)
    parser.add_argument("-C", dest="b_clean_working_directory",
            help="if the argument -C is given, the script (inside the project) resets the project to the downloaded state or (outside the project) deletes the project.",
            action="store_true")
    args = parser.parse_args()

    if args.b_clean_working_directory:
        clean_working_directory()
        sys.exit()

    if args.s_python_version:
        PYTHON_COMMAND = args.s_python_version

    log_file = os.path.abspath(__file__).replace("reproduce_paper.py", "reproduce_paper.log")
    reproduced_main_pfd_file = os.path.abspath(__file__).replace("reproduce_paper.py", "main.pdf")
    print_to_std_and_file("======================== Reproduce '" + PROJECT_NAME + "'s Experiments ========================")
    print_with_timestamp("The script log is at " + str(log_file))

    verify_requirements()
    # project_root = download_repository()
    project_root = SCRIPT_ABS_DIR + "/"

    os.chdir(project_root)

    # create_synthetic_data()
    if args.b_download_dataset:
        sys.exit()

    # Run the experiments
    try:
        local_result_dir = os.getcwd() + "/experiments/results"
        try:
            os.makedirs(local_result_dir)
            print_to_std_and_file("'" + local_result_dir + "' created.")
        except FileExistsError:
            pass
        if not args.b_run_no_experiment:
            functions = {# "[Name]" : "[Runtime]"
                    "support" : "60 seconds (00:00:01 h)",
                    "update_experiments" : "43.200 seconds (12:00:00 h)",
                    
                    }
            # run_ilm = ""
            # if args.b_run_extended_il_miner:
            #     run_ilm = "--ilm"
            exp_args = ["", ""]
            assert len(functions) == len(exp_args)
            try:
                run_experiments(local_result_dir, PYTHON_COMMAND, functions.keys(), "experiments/", ".verification_file", args=exp_args, estimated_runtimes=list(functions.values()))
            except AssertionError:
                print_to_std_and_file("Not all experiments without an error! Check the log for more details!")

        # Compile .tex to .pdf
        print_to_std_and_file("Compiling the paper.pdf...")
        compile_main_pdf()
    finally:
        os.chdir(project_root)
