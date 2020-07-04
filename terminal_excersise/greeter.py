
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


version = "0.0.1"

if len(sys.argv) == 1:
    try:
        from PyInquirer import (Token, ValidationError,
                                Validator, print_json, prompt, style_from_dict)
        pyq = True
    except ImportError:
        pyq = False
    try:
        from pyfiglet import figlet_format
        pyfg = True
    except ImportError:
        pyfg = False
    try:
        import colorama
        colorama.init()
        colorama = True
    except ImportError:
        colorama = None
    try:
        import flatlatex
        c = flatlatex.converter()
        latex = True
    except ImportError:
        latex = False
    try:
        from termcolor import colored
    except ImportError:
        colored = None

elif sys.argv[1] == '-i':
    try:
        from PyInquirer import (Token, ValidationError,
                                Validator, print_json, prompt, style_from_dict)
        pyq = True
    except ImportError:
        install('PyInquirer')
        try:
            from PyInquirer import (
                Token, ValidationError, Validator, print_json, prompt, style_from_dict)
        except ImportError:
            pyq = False
    try:
        import click
    except ImportError:
        install('click')
        try:
            import click
        except ImportError:
            none = True
    try:
        from pyfiglet import figlet_format
        pyfg = True
    except ImportError:
        install('pyfiglet')
        try:
            from pyfiglet import figlet_format
            pyfg = True
        except ImportError:
            pyfg = False
    try:
        import colorama
        colorama.init()
        colorama = True
    except ImportError:
        install('colorama')
        try:
            import colorama
            colorama.init()
            colorama = True
        except ImportError:
            colorama = None
    try:
        import flatlatex
        c = flatlatex.converter()
        latex = True
    except ImportError:
        install('flatlatex')
        try:
            import flatlatex
            c = flatlatex.converter()
            latex = True
        except ImportError:
            latex = False
    try:
        from termcolor import colored
    except ImportError:
        install('termcolor')
        try:
            from termcolor import colored
        except ImportError:
            colored = None

style = style_from_dict({
    Token.QuestionMark: '#fac731 bold',
    Token.Answer: '#4688f1 bold',
    Token.Instruction: '',  # default
    Token.Separator: '#cc5454',
    Token.Selected: '#0abf5b',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Question: '',
})


def title(name='ENR', font='standard', color='blue'):
    if pyfg:
        text = str.strip(figlet_format(name, font=font))
    else:
        text = name
    if colorama:
        print(colored(text, color), end='')
        print(version)
        print()
    else:
        print(text, end='')
        print(version)
        print()


def cConfig(answers):
    if answers['method'] == 'NR':
        return False
    elif answers['span'] is True:
        return False
    return True


def display(answers):
    if answers['method'] == 'ENR' and answers['span'] is True:
        return False
    return True


function_list = [r'x^2'+'\n' + r'\; \,'+r'x ^ 2', r'e ^ {x+1}']
if latex:
    function_list = [c.convert(a) for a in function_list]


def askTestInfo():
    questions = [
        {'type': 'list',
         'name': 'method',
         'message': 'ENR/NR',
         'default': 'ENR',
         'choices': ['ENR', 'NR']},

        {'type': 'list',
         'name': 'func',
         'message': 'Function to test',
         'choices': function_list},

        {'type': 'confirm',
         'name': 'span',
         'message': 'Run across all c configurations',
         'default': False,
         'when': lambda answers: answers['method'] == 'ENR'},

        {'type': 'list',
         'name': 'config',
         'message': 'Choose c configurations',
         'choices': ['2x', 'x+1'],
         'when': cConfig},

        {'type': 'input',
         'name': 'range',
         'message': 'Test range',
         'default': '[-50, 50]'},

        {'type': 'input',
         'name': 'resolution',
         'message': 'Resoultion of result',
         'default': '512'},

        {'type': 'confirm',

         'name': 'show',
         'message': 'Show output',
         'default': 'False',
         'when': display},

        {'type': 'confirm',
         'name': 'save',
         'message': 'Autosave files',
         'default': False,
         'when': display},


    ]
    answers = prompt(questions, style=style)
    return answers


def askTestInfoGen():
    method = input("ENR/NR?")
    func = input("Function to test: "+repr(function_list))

    return


def main():
    title()
    if pyq:
        answers = askTestInfo()
    else:
        answers = askTestInfoGen()


if __name__ == '__main__':
    main()
