from setuptools import find_packages, setup  # Importing necessary functions from setuptools module

HYPEN_E_DOT = '-e .'  # Defining a constant variable

def get_requirements(file_path):
    '''
    This function will return the list of requirements.
    :param file_path: str, path to the requirements file
    :return: list of str, requirements
    '''
    requirements = []  # Initializing an empty list for requirements
    with open(file_path) as file_obj:  # Opening the file specified by file_path
        requirements = file_obj.readlines()  # Reading all lines of the file into a list
        requirements = [req.replace("\n", "") for req in requirements]  # Removing newline characters from each line

        if HYPEN_E_DOT in requirements:  # Checking if HYPEN_E_DOT is present in requirements list
            requirements.remove(HYPEN_E_DOT)  # Removing HYPEN_E_DOT from the requirements list
    
    return requirements  # Returning the list of requirements

setup(
    name='casestudyinterviewprep',  # Setting the name of the project
    version='0.0.1',  # Setting the version of the project
    author='Rosanne',  # Setting the author's name
    author_email='rosanneodiero4@gmail.com',  # Setting the author's email
    packages=find_packages(),  # Finding all packages in the project
    install_requires=get_requirements('requirements.txt')  # Retrieving the list of requirements from the 'requirements.txt' file
)
